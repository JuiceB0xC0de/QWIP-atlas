from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    import orjson

    return orjson.loads(Path(path).read_bytes())


def read_census(path: str | Path) -> list[dict[str, Any]]:
    """Read a census file, transparently handling .json or .npz formats.

    The .npz format stores activation arrays in binary plus a small JSON
    metadata block, which is much smaller and faster than a huge JSON array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".npz":
        return _read_npz_census(p)
    return read_json(p)


def _read_npz_census(path: Path) -> list[dict[str, Any]]:
    import numpy as np
    import orjson

    z = np.load(path, allow_pickle=False)
    metadata = orjson.loads(z["_metadata"].tobytes())
    records = []
    for i, meta in enumerate(metadata):
        rec = dict(meta)
        for key in z.files:
            if key == "_metadata":
                continue
            arr = z[key]
            # Upcast float16 census arrays back to float32 for downstream code.
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            rec[key] = arr[i].tolist()
        records.append(rec)
    return records


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


def write_npz_array_stream(path: str | Path):
    """Context manager that accumulates per-batch arrays and writes a .npz file.

    Each call to write() accepts a list of metadata dicts and a dict of numpy
    arrays for one batch. Per-batch arrays are immediately saved to temporary
    .npy files so memory stays flat; at close time they are concatenated into
    the final compressed .npz.

    Activations are stored as float16 to cut file size and write bandwidth;
    the read path upcasts back to float32.
    """
    import numpy as np
    import orjson

    class _Stream:
        def __init__(self, out: Path):
            self.out = Path(out)
            self.metadata: list[dict[str, Any]] = []
            self.temp_dir = self.out.with_suffix(".tmp")
            self.temp_files: dict[str, list[Path]] = {}
            self.batch_index = 0

        def __enter__(self):
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            return self

        def write(
            self,
            metadata: list[dict[str, Any]],
            arrays: dict[str, Any],
        ) -> None:
            import tempfile

            self.metadata.extend(metadata)
            for k, arr in arrays.items():
                tmp = self.temp_dir / f"{k}_batch{self.batch_index:06d}.npy"
                np.save(tmp, arr)
                self.temp_files.setdefault(k, []).append(tmp)
            self.batch_index += 1

        def __exit__(self, exc_type, exc, tb):
            import shutil

            if exc_type is not None:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                return
            final_arrays: dict[str, Any] = {}
            for k, paths in self.temp_files.items():
                parts = [np.load(p, mmap_mode="r") for p in paths]
                stacked = np.concatenate(parts, axis=0)
                # float16 keeps full dynamic range for activations; halves size.
                if stacked.dtype == np.float32:
                    stacked = stacked.astype(np.float16)
                final_arrays[k] = stacked
            metadata_json = orjson.dumps(self.metadata, default=str)
            final_arrays["_metadata"] = np.frombuffer(metadata_json, dtype=np.uint8)
            np.savez_compressed(self.out, **final_arrays)
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    return _Stream(Path(path))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"
