"""Print (and optionally save) the exact semantic specification and its hash (D15).

    python -m scripts.print_semantic_spec --base-revision <exact frozen-base commit> --out spec.json

The collector writes the printed hash into every record's ``semantic_hash``;
``run_forecasting_eval --semantic-hash <hash>`` then rejects any record made
under different definitions.  Without --base-revision the spec is UNPINNED and
the hash is a preview only.
"""
import argparse
import json

from src.third_eye.forecasting.semantics import SemanticConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-revision", default="UNPINNED")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = SemanticConfig(base_revision=args.base_revision)
    spec = {"semantic_hash": cfg.fingerprint(), "pinned": cfg.pinned, "spec": cfg.to_dict()}
    if not cfg.pinned:
        print("WARNING: base revision not pinned; this hash is a PREVIEW, do not use for real data")
    text = json.dumps(spec, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
