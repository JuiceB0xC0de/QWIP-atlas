from __future__ import annotations

import argparse
import os
from pathlib import Path

from qwip_atlas.config import AtlasRunConfig, CorpusSpec, ModelSpec
from qwip_atlas.layers import parse_layer_spec


def _extract_local(args: argparse.Namespace) -> None:
    cfg = AtlasRunConfig(
        model=ModelSpec(
            model_id=args.model,
            revision=args.revision,
            trust_remote_code=not args.no_trust_remote_code,
            dtype=args.dtype,
            device_map=args.device_map,
            max_length=args.max_length,
        ),
        corpus=CorpusSpec(
            path=Path(args.corpus),
            prompt_key=args.prompt_key,
            category_key=args.category_key,
            bucket_key=args.bucket_key,
        ),
        layers=parse_layer_spec(args.layers),
        outdir=Path(args.outdir),
        batch_size=args.batch_size,
        components=set(args.components.split(",")) if args.components else AtlasRunConfig.__dataclass_fields__["components"].default_factory(),
        truncate_to_deepest_layer=not args.no_truncate,
    )
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    from qwip_atlas.extractors import run_local_census

    run_local_census(cfg, hf_token=token)


def main() -> None:
    parser = argparse.ArgumentParser(prog="qwip-atlas")
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser("extract-local", help="Run local HF activation census extraction")
    extract.add_argument("--model", required=True, help="HF model id or local model path")
    extract.add_argument("--revision", default=None)
    extract.add_argument("--corpus", required=True, help="Local JSONL corpus")
    extract.add_argument("--layers", required=True, help="Layer spec, e.g. 0,4,10-12")
    extract.add_argument("--outdir", required=True)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--max-length", type=int, default=512)
    extract.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    extract.add_argument("--device-map", default="auto", help="Transformers device_map; pass '' to disable")
    extract.add_argument("--components", default=None, help="Comma-separated subset: mlp,gate,up,attn,heads,q,k,v")
    extract.add_argument("--prompt-key", default="prompt")
    extract.add_argument("--category-key", default="category")
    extract.add_argument("--bucket-key", default="bucket")
    extract.add_argument("--hf-token", default=None)
    extract.add_argument("--no-trust-remote-code", action="store_true")
    extract.add_argument("--no-truncate", action="store_true")
    extract.set_defaults(func=_extract_local)

    args = parser.parse_args()
    if getattr(args, "device_map", None) == "":
        args.device_map = None
    args.func(args)


if __name__ == "__main__":
    main()
