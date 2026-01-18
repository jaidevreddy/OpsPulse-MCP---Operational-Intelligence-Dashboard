import os
import json
from pathlib import Path
from typing import Any, Dict, Tuple

# Allow root override (useful in Docker)
ROOT = Path(os.getenv("OPSPULSE_ROOT", Path(__file__).resolve().parents[1]))

_cache: Dict[Tuple[str, str], Any] = {}


def file_sig(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_mtime_ns}:{st.st_size}"


def load_json_cached(path: Path) -> Any:
    key = ("json", file_sig(path))
    if key in _cache:
        return _cache[key]
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    _cache[key] = obj
    return obj


def memo_get(key: Tuple[str, str]) -> Any:
    return _cache.get(key)


def memo_set(key: Tuple[str, str], value: Any) -> None:
    _cache[key] = value


def clear_cache() -> None:
    _cache.clear()
