"""Fire detection image saver with rate limiting."""

import time
import threading
from pathlib import Path
from typing import List, Optional

from django.conf import settings


class FireImageSaver:
    """
    Thread-safe fire image saver with rate limiting.
    
    Features:
        - Automatic directory creation
        - Rate limiting to prevent spam
        - Unique filename generation with millisecond precision
        - Thread-safe save operations
    """

    def __init__(self):
        """Initialize saver with configuration from Django settings."""
        self.base_dir = Path(
            getattr(settings, 'FIRE_SAVE_DIR', Path(settings.BASE_DIR) / 'media' / 'fire')
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_interval = float(getattr(settings, 'FIRE_SAVE_MIN_INTERVAL', 2.0))
        self.max_files = int(getattr(settings, 'FIRE_SAVE_MAX', 1000))
        
        self._last_save_ts = 0.0
        self._save_lock = threading.Lock()

    def save(self, jpg_bytes: bytes) -> Optional[Path]:
        """
        Save fire detection image with rate limiting.
        
        Args:
            jpg_bytes: JPEG image data
            
        Returns:
            Path to saved file, or None if skipped due to rate limit
        """
        with self._save_lock:
            now = time.time()
            
            # Rate limiting
            if now - self._last_save_ts < self.min_interval:
                return None
            
            # Generate unique filename
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            milliseconds = int((now - int(now)) * 1000)
            filename = f'fire_{timestamp}_{milliseconds:03d}.jpg'
            file_path = self.base_dir / filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(jpg_bytes)
            
            self._last_save_ts = now
            
            # Cleanup old files if needed
            self._cleanup_old_files()
            
            return file_path

    def _cleanup_old_files(self) -> None:
        """Remove oldest files if exceeding max_files limit."""
        files = self.list_files(limit=None)
        if len(files) > self.max_files:
            for old_file in files[self.max_files:]:
                try:
                    old_file.unlink()
                except Exception:
                    pass

    def list_files(self, limit: Optional[int] = 50) -> List[Path]:
        """
        List saved fire images sorted by modification time (newest first).
        
        Args:
            limit: Maximum number of files to return, None for all
            
        Returns:
            List of file paths
        """
        files = sorted(
            self.base_dir.glob('*.jpg'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return files[:limit] if limit else files

    def latest(self) -> Optional[Path]:
        """Get path to most recent fire image."""
        files = self.list_files(limit=1)
        return files[0] if files else None
    
    def count(self) -> int:
        """Get total number of saved images."""
        return len(list(self.base_dir.glob('*.jpg')))
