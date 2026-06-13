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


def finalize_layer(tmp_dir: Path, out_path: Path) -> tuple[Path, bool, str, float]:
    """Concatenate temp .npy files into one compressed .npz."""
    import time

    t0 = time.time()
    try:
        if not tmp_dir.exists():
            return out_path, False, "no tmp dir", time.time() - t0

        npy_files = sorted(tmp_dir.glob("*_batch*.npy"))
        if not npy_files:
            return out_path, False, "no temp files", time.time() - t0

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
            return out_path, False, "missing metadata file", time.time() - t0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **final_arrays)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return out_path, True, f"{len(metadata)} rows", time.time() - t0

    except Exception as exc:
        return out_path, False, str(exc), time.time() - t0


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

    print(f"finalizing {len(tmp_dirs)} layers from {root} with 8 workers")
    print("(each dot is one layer completed)")

    from tqdm import tqdm

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(finalize_layer, tmp_dir, tmp_dir.with_suffix("")): tmp_dir for tmp_dir in tmp_dirs}

        with tqdm(total=len(tmp_dirs), unit="layer", desc="finalize") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                out_path, ok, msg, elapsed = fut.result()
                status = "OK" if ok else "FAIL"
                pbar.set_postfix({out_path.name: f"{elapsed:.1f}s"})
                pbar.update(1)
                if not ok:
                    pbar.write(f"[FAIL] {out_path.name}: {msg}")
                completed += 1

    print(f"done — {completed}/{len(tmp_dirs)} layers finalized")


if __name__ == "__main__":
    main()
