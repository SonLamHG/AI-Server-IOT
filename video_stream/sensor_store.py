from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Deque, Dict, List, Optional
from django.utils import timezone


@dataclass
class SensorEntry:
    device: str
    smoke: float
    flame: float
    unit: str
    received_at: str  # ISO string

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> Optional['SensorEntry']:
        try:
            device = str(payload.get('device', 'unknown'))
            smoke = float(payload.get('smoke', 0.0))
            flame = float(payload.get('flame', 0.0))
            unit = str(payload.get('unit', '')) or '%'
        except (TypeError, ValueError):
            return None
        return cls(
            device=device,
            smoke=smoke,
            flame=flame,
            unit=unit,
            received_at=timezone.now().isoformat(),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SensorStore:
    def __init__(self, maxlen: int = 3600):
        self._lock = Lock()
        self._buffer: Deque[SensorEntry] = deque(maxlen=maxlen)
        self._latest: Optional[SensorEntry] = None

    def add(self, entry: SensorEntry) -> SensorEntry:
        with self._lock:
            self._buffer.append(entry)
            self._latest = entry
            return entry

    def latest(self) -> Optional[SensorEntry]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest

    def snapshot(self, limit: int = 100) -> List[Dict[str, object]]:
        limit = max(1, limit)
        with self._lock:
            items = list(self._buffer)
        return [e.to_dict() for e in items[-limit:]]


store = SensorStore()
