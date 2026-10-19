#!/usr/bin/env python
"""PhishGuard AI - re-evaluate shipped model artifacts (milestone P2).

    python evaluate_models.py            # human-readable metric table
    python evaluate_models.py --check    # exit 1 if artifact predictions drift
                                          # from committed models/metrics.json

Each artifact is loaded and scored on the exact domain-disjoint test split
rebuilt deterministically from the dataset and recorded seed, so the numbers
below are reproducible on any machine with the committed data.
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

from phishguard.features import SHIPPED_FEATURE_SET, ablation_features  # noqa: E402
from phishguard.ml.data import LABEL_COLUMN  # noqa: E402
from phishguard.ml.metrics import classification_metrics, measure_latency_ms  # noqa: E402
from phishguard.ml.pipeline import build_splits, prepare_data  # noqa: E402
from phishguard.ml.registry import DEFAULT_REGISTRY_PATH, load_registry, sha256_file  # noqa: E402
from phishguard.ml.verify import QUALITY_METRICS, compare_metrics, experiment_index  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true",
        help="compare against committed metrics.json (exit 1 on drift)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6,
        help="max absolute metric drift accepted by --check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not METRICS_PATH.is_file():
        print(f"ERROR: {METRICS_PATH} not found; run train_model.py first", file=sys.stderr)
        return 1
    results = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    registry = load_registry(DEFAULT_REGISTRY_PATH)
    if not registry:
        print("ERROR: registry is empty", file=sys.stderr)
        return 1

    seed = int(results["seed"])
    splits = build_splits(prepare_data(), seed)
    split = splits["domain_disjoint"]
    columns = ablation_features(SHIPPED_FEATURE_SET)
    x_test = split.test[columns]
    y_test = split.test[LABEL_COLUMN]

    committed_index = experiment_index(results)
    recomputed_records: list[dict] = []
    committed_records: list[dict] = []
    problems: list[str] = []

    header = (
        f"{'model':<22} {'accuracy':>8} {'precision':>9} {'recall':>7} "
        f"{'f1':>7} {'roc_auc':>8} {'pr_auc':>7} {'fpr':>7} {'fnr':>7} "
        f"{'size':>10} {'lat_ms':>7}"
    )
    print(header)
    print("-" * len(header))
    for artifact, entry in sorted(registry.items()):
        model_key = Path(artifact).stem
        artifact_path = MODELS_DIR / artifact
        if not artifact_path.is_file():
            problems.append(f"{artifact}: file missing")
            continue
        actual_sha = sha256_file(artifact_path)
        if actual_sha != entry["artifact_sha256"]:
            problems.append(
                f"{artifact}: sha256 drift ({actual_sha[:12]} != "
                f"{entry['artifact_sha256'][:12]})"
            )
            continue

        model = joblib.load(artifact_path)
        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]
        metrics = classification_metrics(y_test, y_pred, y_score)
        latency = measure_latency_ms(model, x_test)
        record = {
            "model": model_key,
            "protocol": "domain_disjoint",
            "feature_set": SHIPPED_FEATURE_SET,
            "metrics": metrics,
        }
        recomputed_records.append(record)

        key = (model_key, "domain_disjoint", SHIPPED_FEATURE_SET)
        if key in committed_index:
            committed_records.append(
                {k: committed_index[key][k] for k in ("model", "protocol", "feature_set", "metrics")}
            )
        else:
            problems.append(f"{artifact}: no committed metrics for {key}")

        m = metrics
        print(
            f"{model_key:<22} {m['accuracy']:>8.4f} {m['precision']:>9.4f} {m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>7.4f} "
            f"{m['fpr']:>7.4f} {m['fnr']:>7.4f} "
            f"{entry['artifact_size_bytes']:>10} {latency['mean_ms']:>7.3f}"
        )

    print(
        f"\nEvaluated {len(recomputed_records)}/{len(registry)} shipped artifacts "
        f"on domain_disjoint test rows (n={len(y_test)}, seed={seed})"
    )

    if args.check:
        diffs = compare_metrics(
            {"experiments": committed_records},
            {"experiments": recomputed_records},
            args.tolerance,
        )
        problems.extend(diffs)
        if problems:
            print("DRIFT DETECTED:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"OK: all quality metrics reproduce within tolerance {args.tolerance:g}")
        for record in recomputed_records:
            missing = [
                metric
                for metric in QUALITY_METRICS
                if record["metrics"].get(metric) is None
            ]
            if missing:
                print(f"  note: {record['model']} has null metrics: {missing}")
    elif problems:
        for problem in problems:
            print(f"warning: {problem}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
