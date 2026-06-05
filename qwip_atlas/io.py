from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    import orjson

    return orjson.loads(Path(path).read_bytes())


def write_json(path: str | Path, obj: Any) -> None:
    import orjson

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    import orjson

    with Path(path).open("rb") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def write_json_array_stream(path: str | Path):
    """Return a tiny context manager for streaming a JSON array incrementally."""

    class _Stream:
        def __init__(self, out: Path):
            self.out = out
            self.handle = None
            self.first = True
            self.count = 0

        def __enter__(self):
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.out.open("wb")
            self.handle.write(b"[")
            return self

        def write(self, obj: Any) -> None:
            import orjson

            if not self.first:
                self.handle.write(b",")
            self.handle.write(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY))
            self.first = False
            self.count += 1

        def __exit__(self, exc_type, exc, tb):
            self.handle.write(b"]")
            self.handle.close()

    return _Stream(Path(path))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"
