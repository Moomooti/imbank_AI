"""Phase 0: Download the CTranslate2-converted NLLB model from HuggingFace.

Model: OpenNMT/nllb-200-distilled-1.3B-ct2-int8
This is the CPU-inference engine swap described in FinHOLLY Phase 0 — the
underlying NLLB weights are unchanged, only the runtime engine differs
(CTranslate2 instead of vanilla transformers), giving ~3-4x CPU speedup.
"""
import argparse
import sys

from huggingface_hub import snapshot_download

MODEL_ID = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="models/nllb-200-distilled-1.3B-ct2-int8",
        help="Local directory to save the model into",
    )
    args = parser.parse_args()

    print(f"Downloading {MODEL_ID} -> {args.out_dir}")
    path = snapshot_download(repo_id=MODEL_ID, local_dir=args.out_dir)
    print(f"Done. Model saved at: {path}")


if __name__ == "__main__":
    sys.exit(main())
