#!/usr/bin/env python3
import argparse
from pathlib import Path

from transformers import AutoTokenizer, EsmModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        default="facebook/esm2_t33_650M_UR50D",
        help="Hugging Face model repository ID",
    )
    parser.add_argument(
        "--output_dir",
        default="./model/esm2_650m",
        help="Local directory used by --model_locate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer from {args.repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id)
    tokenizer.save_pretrained(output_dir)

    print(f"Downloading model from {args.repo_id}")
    model = EsmModel.from_pretrained(args.repo_id)
    model.save_pretrained(output_dir)

    required = [output_dir / "config.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete model directory; missing: {missing}")

    print(f"ESM2 saved to: {output_dir}")


if __name__ == "__main__":
    main()
