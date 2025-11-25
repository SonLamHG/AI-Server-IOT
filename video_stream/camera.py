"""
Video stream frame grabber with async AI detection.

Architecture:
    - Thread 1 (Frame Reader): Continuously reads frames from source
    - Thread 2 (AI Detector): Processes frames asynchronously for fire detection
    - Non-blocking design ensures smooth video stream regardless of AI processing time
"""

import threading
import time
from typing import Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from django.conf import settings

from modelAI.detectFire import DetectFire
from .saver import FireImageSaver


@dataclass
class VideoConfig:
    """Video stream configuration."""
    source: str
    max_width: Optional[int]
    target_fps: Optional[int]
    jpeg_quality: int
    buffer_size: int

    @classmethod
    def from_settings(cls, source: Optional[str] = None) -> 'VideoConfig':
        """Load configuration from Django settings."""
        return cls(
            source=source or getattr(settings, 'VIDEO_SOURCE_URL', 0),
            max_width=getattr(settings, 'VIDEO_MAX_WIDTH', None),
            target_fps=getattr(settings, 'VIDEO_TARGET_FPS', None),
            jpeg_quality=getattr(settings, 'VIDEO_JPEG_QUALITY', 90),
            buffer_size=getattr(settings, 'VIDEO_BUFFER_SIZE', 5),
        )


class FrameGrabber:
    """
    Multi-threaded video frame grabber with async AI detection.
    
    Features:
        - Singleton pattern for shared instance
        - Non-blocking frame reading
        - Async fire detection in separate thread
        - Auto-reconnection on failures
        - Support for multiple video sources (HTTP, RTSP, webcam, file)
    """

    _instance: Optional['FrameGrabber'] = None
    _instance_lock = threading.Lock()

    def __init__(self, source: Optional[str] = None):
        """Initialize frame grabber with optional source override."""
        self.config = VideoConfig.from_settings(source)
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Frame storage with thread safety
        self._frame_lock = threading.Lock()
        self._frame_ndarr: Optional[np.ndarray] = None  # Current display frame
        self._frame_jpg: Optional[bytes] = None  # Cached JPEG bytes
        self._raw_frame: Optional[np.ndarray] = None  # Frame for AI processing
        
        # Thread management
        self._stopped = threading.Event()
        self._reader_thread = threading.Thread(target=self._read_frames, daemon=True, name="FrameReader")
        self._ai_thread = threading.Thread(target=self._detect_fire, daemon=True, name="AIDetector")
        
        # Fire detection saver
        self._saver = FireImageSaver()

    @classmethod
    def get_instance(cls, source: Optional[str] = None) -> 'FrameGrabber':
        """Get or create singleton instance."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = FrameGrabber(source)
                cls._instance.start()
            return cls._instance

    def start(self) -> None:
        """Start frame reader and AI detection threads."""
        if not self._reader_thread.is_alive():
            self._stopped.clear()
            self._reader_thread.start()
            self._ai_thread.start()

    def stop(self) -> None:
        """Stop all threads and release video capture."""
        self._stopped.set()
        self._release_capture()

    def _release_capture(self) -> None:
        """Safely release video capture."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            finally:
                self.cap = None

    def _open_capture(self) -> bool:
        """
        Open video capture with configuration.
        
        Returns:
            True if capture opened successfully, False otherwise.
        """
        self._release_capture()
        
        self.cap = cv2.VideoCapture(self.config.source)
        if not self.cap.isOpened():
            return False
        
        # Configure buffer size
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        except Exception:
            pass
        
        return True

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if max width is configured."""
        if self.config.max_width and frame.shape[1] > self.config.max_width:
            scale = self.config.max_width / frame.shape[1]
            return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return frame

    def _read_frames(self) -> None:
        """
        Main frame reading loop (Thread 1).
        
        Continuously reads frames from video source and updates shared frame buffer.
        Runs in dedicated thread to ensure non-blocking operation.
        """
        backoff = 0.5
        max_backoff = 5.0
        
        while not self._stopped.is_set():
            # Open capture if not available
            if self.cap is None or not self.cap.isOpened():
                if not self._open_capture():
                    time.sleep(min(backoff, max_backoff))
                    backoff = min(backoff * 2, max_backoff)
                    continue
                backoff = 0.5

            # Read frame from source
            success, frame = self.cap.read()
            if not success or frame is None:
                self._release_capture()
                time.sleep(0.2)
                continue

            # Apply resize if configured
            frame = self._resize_frame(frame)

            # Update shared frame buffer (non-blocking for display)
            with self._frame_lock:
                self._frame_ndarr = frame.copy()
                self._frame_jpg = None  # Invalidate cached JPEG
                self._raw_frame = frame  # For AI processing

            # Throttle FPS if configured
            if self.config.target_fps and self.config.target_fps > 0:
                time.sleep(1.0 / self.config.target_fps)

    def _encode_frame_to_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode frame to JPEG bytes with configured quality."""
        success, buffer = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        return buffer.tobytes() if success else None

    def get_frame(self) -> Optional[bytes]:
        """
        Get latest frame as JPEG bytes.
        
        Returns cached JPEG if available, otherwise encodes current frame.
        Thread-safe operation.
        """
        with self._frame_lock:
            # Return cached JPEG if available
            if self._frame_jpg is not None:
                return self._frame_jpg
            
            # No frame available
            if self._frame_ndarr is None:
                return None
            
            # Encode and cache
            self._frame_jpg = self._encode_frame_to_jpeg(self._frame_ndarr)
            return self._frame_jpg

    def get_frame_ndarray(self) -> Optional[np.ndarray]:
        """
        Get latest frame as numpy array (BGR format).
        
        Returns:
            Copy of current frame or None if no frame available.
        """
        with self._frame_lock:
            if self._frame_ndarr is not None:
                return self._frame_ndarr.copy()
            
            # Fallback: decode from JPEG if available
            if self._frame_jpg is not None:
                return cv2.imdecode(
                    np.frombuffer(self._frame_jpg, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
            
            return None

    def generate_mjpeg_stream(self):
        """
        Generator for MJPEG stream (multipart/x-mixed-replace).
        
        Yields JPEG frames with proper HTTP multipart boundaries.
        Used for streaming video over HTTP.
        """
        boundary = b'frame'
        
        while not self._stopped.is_set():
            frame_data = self.get_frame()
            
            if frame_data is None:
                time.sleep(0.05)
                continue
            
            yield (
                b'--' + boundary + b'\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
                + frame_data + b'\r\n'
            )

    def _process_fire_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Run fire detection on frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, has_fire_detected)
        """
        try:
            processed, has_fire, _ = DetectFire(frame)
            return (processed if processed is not None else frame, has_fire)
        except Exception:
            return (frame, False)

    def _detect_fire(self) -> None:
        """
        Fire detection loop (Thread 2).
        
        Continuously processes frames for fire detection without blocking
        the main frame reading thread. Saves detected fire images.
        """
        last_frame_id = None
        
        while not self._stopped.is_set():
            try:
                # Get current raw frame
                with self._frame_lock:
                    frame = self._raw_frame
                    frame_id = id(frame)
                
                # Skip if no new frame
                if frame is None or frame_id == last_frame_id:
                    time.sleep(0.05)
                    continue
                
                last_frame_id = frame_id
                frame_copy = frame.copy()
                
                # Process fire detection (may take 200-500ms)
                processed, has_fire = self._process_fire_detection(frame_copy)
                
                # Save image if fire detected
                if has_fire:
                    jpg_data = self._encode_frame_to_jpeg(processed)
                    if jpg_data:
                        self._saver.save(jpg_data)
                
            except Exception as e:
                # Log error but continue processing
                time.sleep(0.1)
