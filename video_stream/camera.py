import threading
import time
from typing import Optional

import cv2
import requests
import numpy as np
from django.conf import settings
from modelAI.detectFire import DetectFire
from .saver import FireImageSaver


class FrameGrabber:
    """
    Simple background frame grabber using OpenCV or HTTP MJPEG.
    - Reads frames in a dedicated thread
    - Stores the latest frame in memory with a lock
    - Auto-reconnects on failures
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, source: Optional[str] = None):
        self.source = source or getattr(settings, 'VIDEO_SOURCE_URL', 0)
        self.cap = None
        self._frame_lock = threading.Lock()
        self._frame_ndarr = None  # OpenCV ndarray
        self._frame_jpg = None  # JPEG bytes
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._saver = FireImageSaver()

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = FrameGrabber()
                cls._instance.start()
            return cls._instance

    def start(self):
        if not self._thread.is_alive():
            self._stopped.clear()
            self._thread.start()

    def stop(self):
        self._stopped.set()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def _open(self):
        # Release any previous capture
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        # Open new capture
        self.cap = cv2.VideoCapture(self.source)
        # Try to set a reasonable buffer if supported
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

    def _run(self):
        # If HTTP URL is provided, try MJPEG stream parsing; otherwise use VideoCapture
        if isinstance(self.source, str) and self.source.startswith('http'):
            self._run_http_mjpeg()
        else:
            self._run_videocap()

    def _run_videocap(self):
        backoff = 0.5
        while not self._stopped.is_set():
            if self.cap is None or not self.cap.isOpened():
                self._open()
                if not self.cap.isOpened():
                    time.sleep(min(backoff, 5))
                    backoff = min(backoff * 2, 5)
                    continue
                backoff = 0.5

            ok, frame = self.cap.read()
            if not ok or frame is None:
                # Reopen on failure
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(0.2)
                continue

            # optional resize
            max_w = getattr(settings, 'VIDEO_MAX_WIDTH', None)
            if max_w and frame.shape[1] > max_w:
                scale = max_w / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # chạy detect trên frame đã đọc
            processed, has_fire = self._detect_and_save(frame)

            with self._frame_lock:
                self._frame_ndarr = processed
                self._frame_jpg = None

            # Throttle read rate if a target FPS is desired
            target_fps = getattr(settings, 'VIDEO_TARGET_FPS', None)
            if target_fps and target_fps > 0:
                time.sleep(max(0, (1.0 / target_fps)))

    def _run_http_mjpeg(self):
        # Parse multipart/x-mixed-replace stream
        backoff = 0.5
        session = requests.Session()
        while not self._stopped.is_set():
            try:
                with session.get(self.source, stream=True, timeout=10) as resp:
                    resp.raise_for_status()
                    bytes_buf = b''
                    for chunk in resp.iter_content(chunk_size=4096):
                        if self._stopped.is_set():
                            return
                        if not chunk:
                            continue
                        bytes_buf += chunk
                        # Look for JPEG start and end markers
                        a = bytes_buf.find(b'\xff\xd8')
                        b = bytes_buf.find(b'\xff\xd9')
                        if a != -1 and b != -1 and b > a:
                            jpg = bytes_buf[a:b+2]
                            bytes_buf = bytes_buf[b+2:]
                            arr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            processed = None
                            if arr is not None:
                                # optional resize
                                max_w = getattr(settings, 'VIDEO_MAX_WIDTH', None)
                                if max_w and arr.shape[1] > max_w:
                                    scale = max_w / arr.shape[1]
                                    arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                                processed, has_fire = self._detect_and_save(arr)
                                ok, buf = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), getattr(settings, 'VIDEO_JPEG_QUALITY', 80)])
                                if ok:
                                    jpg = buf.tobytes()
                            with self._frame_lock:
                                self._frame_jpg = jpg
                                self._frame_ndarr = processed

                            target_fps = getattr(settings, 'VIDEO_TARGET_FPS', None)
                            if target_fps and target_fps > 0:
                                time.sleep(max(0, (1.0 / target_fps)))

                # If we exit the with block, reconnect
                time.sleep(min(backoff, 5))
                backoff = min(backoff * 2, 5)
            except Exception:
                time.sleep(min(backoff, 5))
                backoff = min(backoff * 2, 5)

    def get_frame(self) -> Optional[bytes]:
        """Return latest frame encoded as JPEG bytes."""
        with self._frame_lock:
            if self._frame_jpg is not None:
                return self._frame_jpg
            if self._frame_ndarr is None:
                return None
            ok, buf = cv2.imencode(
                '.jpg',
                self._frame_ndarr,
                [int(cv2.IMWRITE_JPEG_QUALITY), getattr(settings, 'VIDEO_JPEG_QUALITY', 80)],
            )
            if not ok:
                return None
            self._frame_jpg = buf.tobytes()
            return self._frame_jpg

    def get_frame_ndarray(self) -> Optional[np.ndarray]:
        """Return latest frame as numpy ndarray (BGR)."""
        with self._frame_lock:
            if self._frame_ndarr is not None:
                return self._frame_ndarr.copy()
            if self._frame_jpg is None:
                return None
            # Decode JPG into ndarray
            arr = cv2.imdecode(np.frombuffer(self._frame_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return None
            return arr

    def mjpeg_generator(self):
        boundary = 'frame'
        while not self._stopped.is_set():
            data = self.get_frame()
            if data is None:
                time.sleep(0.05)
                continue
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
            )

    def _detect_and_save(self, frame: np.ndarray):
        """Run DetectFire on frame, save if fire detected, return processed frame and flag."""
        try:
            processed, has_fire, _ = DetectFire(frame)
        except Exception:
            processed, has_fire = frame, False
        if processed is None:
            processed = frame
        if has_fire:
            try:
                ok, buf = cv2.imencode(
                    '.jpg',
                    processed,
                    [int(cv2.IMWRITE_JPEG_QUALITY), getattr(settings, 'VIDEO_JPEG_QUALITY', 80)],
                )
                if ok:
                    self._saver.save(buf.tobytes())
            except Exception:
                pass
        return processed, has_fire
