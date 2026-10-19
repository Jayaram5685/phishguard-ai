#!/usr/bin/env python
"""PhishGuard AI - reproducible training CLI (milestone P2).

    python train_model.py             # full benchmark; writes models/*.joblib,
                                      # models/registry.json, models/metrics.json
    python train_model.py --check     # re-run everything and verify the committed
                                      # metrics reproduce (exit 1 on drift)

Deterministic for a fixed ``--seed`` (default 42) and the committed dataset.
Methodology: docs/METHODOLOGY.md. Evaluation protocol rationale:
"why domain-disjoint matters" is covered there too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402

from phishguard.features import SHIPPED_FEATURE_SET  # noqa: E402
from phishguard.ml.models import MODEL_SPECS  # noqa: E402
from phishguard.ml.pipeline import run_experiments  # noqa: E402
from phishguard.ml.registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    build_registry_entry,
    sha256_file,
    update_registry,
)
from phishguard.ml.verify import compare_metrics, experiment_index  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

#: Artifacts shipped to models/ and served by the API. Each model must earn
#: its slot on the primary protocol (domain_disjoint + full features) --
#: results live in models/metrics.json.
SHIPPED_MODEL_KEYS = ("logistic_regression", "random_forest", "xgboost", "lightgbm")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42, help="split/training seed (default 42)")
    parser.add_argument(
        "--tolerance", type=float, default=1e-6,
        help="max absolute metric drift accepted by --check",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="verify committed metrics.json instead of writing artifacts",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-experiment progress")
    return parser.parse_args(argv)


def _print_primary_table(results: dict) -> None:
    index = experiment_index(results)
    header = (
        f"{'model':<22} {'accuracy':>8} {'precision':>9} {'recall':>7} "
        f"{'f1':>7} {'roc_auc':>8} {'pr_auc':>7} {'fpr':>7} {'fnr':>7}"
    )
    print(f"\nPrimary protocol: domain_disjoint, {SHIPPED_FEATURE_SET} feature set")
    print(header)
    print("-" * len(header))
    for key in sorted(index):
        model, protocol, feature_set = key
        if protocol != "domain_disjoint" or feature_set != SHIPPED_FEATURE_SET:
            continue
        m = index[key]["metrics"]
        print(
            f"{model:<22} {m['accuracy']:>8.4f} {m['precision']:>9.4f} {m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>7.4f} "
            f"{m['fpr']:>7.4f} {m['fnr']:>7.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = None if args.quiet else lambda msg: print(msg, flush=True)
    results, models = run_experiments(seed=args.seed, keep_models=True, progress=progress)

    if args.check:
        if not METRICS_PATH.is_file():
            print(f"ERROR: {METRICS_PATH} not found; run train_model.py first", file=sys.stderr)
            return 1
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        diffs = compare_metrics(committed, results, args.tolerance)
        if diffs:
            print("METRIC DRIFT DETECTED:", file=sys.stderr)
            for diff in diffs:
                print(f"  {diff}", file=sys.stderr)
            return 1
        print(
            f"OK: {len(results['experiments'])} experiments reproduced "
            f"within tolerance {args.tolerance:g} (seed={args.seed})"
        )
        return 0

    MODELS_DIR.mkdir(exist_ok=True)
    shipped_entries: dict[str, dict] = {}
    shipped_block: dict[str, dict] = {}
    index = experiment_index(results)
    for model_key in SHIPPED_MODEL_KEYS:
        spec = MODEL_SPECS[model_key]
        artifact_path = MODELS_DIR / spec.artifact
        estimator = models.get(model_key)
        if estimator is None:
            print(f"ERROR: model {model_key} was not fitted", file=sys.stderr)
            return 1
        joblib.dump(estimator, artifact_path)
        experiment = index[(model_key, "domain_disjoint", SHIPPED_FEATURE_SET)]
        shipped_entries[spec.artifact] = build_registry_entry(
            model_key, artifact_path, results, experiment, args.seed
        )
        shipped_block[spec.artifact] = {
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        }

    results["shipped_artifacts"] = shipped_block
    METRICS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    update_registry(shipped_entries, DEFAULT_REGISTRY_PATH)

    _print_primary_table(results)
    print(
        f"\nWrote {len(shipped_entries)} artifacts -> {MODELS_DIR}, "
        f"registry -> {DEFAULT_REGISTRY_PATH}, metrics -> {METRICS_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
