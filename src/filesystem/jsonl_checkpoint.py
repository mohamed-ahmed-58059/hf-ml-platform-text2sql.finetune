import json
import threading
from pathlib import Path
from typing import Optional


class JsonlCheckpoint:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._file = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path or not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, row: dict) -> None:
        if not self.path:
            return
        with self._lock:
            if self._file is None:
                self._file = self.path.open("a", encoding="utf-8")
            self._file.write(json.dumps(row) + "\n")
            self._file.flush()
