import os
import time
from pathlib import Path
from typing import List, Optional

from django.conf import settings


class FireImageSaver:
    def __init__(self):
        self.base_dir = Path(getattr(settings, 'FIRE_SAVE_DIR', Path(settings.BASE_DIR) / 'media' / 'fire'))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._last_save_ts = 0.0
        self.min_interval = float(getattr(settings, 'FIRE_SAVE_MIN_INTERVAL', 2.0))

    def save(self, jpg_bytes: bytes) -> Path:
        now = time.time()
        if now - self._last_save_ts < self.min_interval:
            # skip if saved too recently, but still return last file if exists
            latest = self.latest()
            if latest:
                return latest
        ts = time.strftime('%Y%m%d_%H%M%S')
        # ensure unique by adding milliseconds
        ms = int((now - int(now)) * 1000)
        file_path = self.base_dir / f'fire_{ts}_{ms:03d}.jpg'
        with open(file_path, 'wb') as f:
            f.write(jpg_bytes)
        self._last_save_ts = now
        return file_path

    def list_files(self, limit: int = 50) -> List[Path]:
        files = sorted(self.base_dir.glob('*.jpg'), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[:limit]

    def latest(self) -> Optional[Path]:
        files = self.list_files(limit=1)
        return files[0] if files else None
