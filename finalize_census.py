#!/usr/bin/env python3
"""Finalize partially-written census .npz files from temp .npy batches.

Usage:
    python finalize_census.py runs/llama-3-8b/census-v4

This is useful when the main extractor has finished all forward passes but is
still serializing the final .npz files one layer at a time. It reads the
existing temp .npy files, concatenates them, and writes the compressed .npz
for every incomplete layer in parallel.
"""
from __future__ import annotations

import concurrent.futures
import shutil
import sys
from pathlib import Path

import numpy as np
import orjson


def finalize_layer(tmp_dir: Path, out_path: Path) -> tuple[Path, bool, str]:
    """Concatenate temp .npy files into one compressed .npz."""
    try:
        if not tmp_dir.exists():
            return out_path, False, "no tmp dir"

        npy_files = sorted(tmp_dir.glob("*_batch*.npy"))
        if not npy_files:
            return out_path, False, "no temp files"

        # Group files by field key.
        field_files: dict[str, list[Path]] = {}
        for p in npy_files:
            key = p.stem.rsplit("_batch", 1)[0]
            field_files.setdefault(key, []).append(p)

        final_arrays: dict[str, np.ndarray] = {}
        metadata = None
        for key, paths in field_files.items():
            if key == "_metadata":
                # Metadata is a single JSON array stored as a .npy uint8 buffer.
                meta_arr = np.load(paths[0])
                metadata = orjson.loads(meta_arr.tobytes())
                final_arrays["_metadata"] = meta_arr
                continue
            parts = [np.load(p, mmap_mode="r") for p in sorted(paths)]
            stacked = np.concatenate(parts, axis=0)
            if stacked.dtype == np.float32:
                stacked = stacked.astype(np.float16)
            final_arrays[key] = stacked

        if metadata is None:
            return out_path, False, "missing metadata file"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **final_arrays)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return out_path, True, f"{len(metadata)} rows"

    except Exception as exc:
        return out_path, False, str(exc)


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <census-dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    tmp_dirs = [d for d in root.iterdir() if d.is_dir() and d.suffix == ".tmp"]
    if not tmp_dirs:
        print("no .tmp directories found; nothing to finalize")
        return

    print(f"finalizing {len(tmp_dirs)} layers from {root}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for tmp_dir in tmp_dirs:
            out_path = tmp_dir.with_suffix("")
            futures.append(pool.submit(finalize_layer, tmp_dir, out_path))

        for fut in concurrent.futures.as_completed(futures):
            out_path, ok, msg = fut.result()
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {out_path.name}: {msg}")

    print("done")


if __name__ == "__main__":
    main()
