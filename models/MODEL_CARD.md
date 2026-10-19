# Model Card — PhishGuard AI

> **Status: CURRENT (milestone P2).** All four artifacts below were trained in
> this repository by `python train_model.py --seed 42` on the leakage-free
> `domain_disjoint` protocol, and every metric was reproduced from scratch by
> `python evaluate_models.py --check`. Methodology and honest reading of the
> numbers: [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md).

## Artifacts

| File | Algorithm | Size | SHA-256 |
|---|---|---|---|
| `models/random_forest.joblib` | `RandomForestClassifier` (500 trees, `class_weight=balanced`) | 13,453,865 B | `7c336ebfd14e0b1b35397ac632ea6acb39efc362826dc58b031d4e60bfa5afb3` |
| `models/xgboost.joblib` | `XGBClassifier` (500, depth 8, lr 0.02, `hist`) | 933,054 B | `a5a7ffefb6cda8ce51aa6fe49024455af8495789f9e66b852e5bcf7d700f3c8a` |
| `models/lightgbm.joblib` | `LGBMClassifier` (500, lr 0.02, 31 leaves, deterministic) | 1,707,828 B | `bd8be6d03ba06faad8d69fbc2928f32b41653cda831714747537d245d3b147a1` |
| `models/logistic_regression.joblib` | `LogisticRegression` (lbfgs, `class_weight=balanced`) | 1,439 B | `ebfcd68d090e2f25c886f1e9e32998077a9b59101360ab27150eeef42ffa7044` |

Machine-readable provenance (hyperparameters, split, metrics, library
versions, checksums) lives in [`models/registry.json`](registry.json);
the full 24-experiment benchmark lives in
[`models/metrics.json`](metrics.json).

## Intended use and limits

* **Use:** real-time triage of a user-supplied URL as one signal among many —
  the response carries the hard label, the model's own `score`
  (`P(phishing)`, uncalibrated) and feature-level `reasons`.
* **Not a block list.** On the primary protocol the shipped models miss 27–34%
  of phishing URLs and flag 7.5–15% of legitimate ones. Never present the
  verdict as certainty.
* **Out of scope:** adversarial inputs, brand/certificate intelligence,
  automated enforcement decisions.

## Training data

* **Dataset:** `DataFiles/5.urldata.csv`, SHA-256
  `5c74254ce576a730beaea2ed15c3acbfd7e02dad853b552d248b5af5a034b188`
  — 10,000 rows, 3,553 domains, 581 distinct feature vectors (features-v2).
* **Pool:** deduplicated + mixed-label domains dropped → **4,366 rows**
  (524 legitimate / 3,842 phishing).
* **Split:** `domain_disjoint`, seed 42, 80/20 — train 3,488 rows / 2,839
  domains, test 878 rows / 711 domains, **domain overlap = 0**.
* **Provenance:** unverified upstream (collection dates and licence were never
  recorded) — see [`DataFiles/DATASETS.md`](../DataFiles/DATASETS.md).

## Feature schema (input contract)

Exactly these 15 columns, in this order, as numeric values
(features-v2, fingerprint `f0099fbf914b0fe9…`):

```
Have_IP, Have_At, URL_Length, URL_Depth, Redirection, https_Domain,
TinyURL, Prefix/Suffix, DNS_Record, Domain_Age, Domain_End,
iFrame, Mouse_Over, Right_Click, Web_Forwards
```

Binary columns are `0/1` except `URL_Depth` (count).
`label_semantics`: `0 = legitimate`, `1 = phishing`.

Changes vs the legacy 16-column contract:

* **`Web_Traffic` removed** — its Alexa source was discontinued in 2022 and
  cannot be served. Forcing it to the only available value costs **3.8–4.4
  accuracy points** (measured, [`docs/METHODOLOGY.md` §6.2](../docs/METHODOLOGY.md)).
* **`Have_IP` is hostname-scoped** — the legacy whole-URL check could never
  succeed and was dead in serving.
* The four page features are best-effort and ambiguous by construction
  (`1` can mean "pattern found" *or* "fetch failed"); the API reports which
  signals were unavailable instead of hiding it.

`registry.json → feature_columns` must equal this schema exactly; the app
refuses to boot otherwise (asserted at import).

## Performance

Primary protocol `domain_disjoint`, features-v2, n = 878 test rows
(772 phishing / 106 legitimate), seed 42:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | p95 latency |
|---|---|---|---|---|---|---|---|---|---|
| `logistic_regression` | 0.6948 | 0.9846 | 0.6632 | 0.7926 | **0.8753** | **0.9814** | **0.0755** | 0.3368 | 6.7 ms |
| `random_forest` | **0.7472** | 0.9725 | **0.7332** | **0.8360** | 0.8496 | 0.9744 | 0.1509 | 0.2668 | 108.3 ms |
| `xgboost` | 0.7255 | 0.9801 | 0.7021 | 0.8181 | 0.8632 | 0.9796 | 0.1038 | 0.2979 | 14.5 ms |
| `lightgbm` | 0.7244 | 0.9749 | 0.7047 | 0.8180 | 0.8598 | 0.9790 | 0.1321 | 0.2953 | 5.7 ms |

95% Wilson intervals for every metric are in `registry.json → evaluation.metrics.ci95`
(e.g. Random Forest accuracy CI **[0.7174, 0.7748]**).

**Read these numbers with the baselines:** the test split is 87.9% phishing,
so "always say phishing" scores 0.8793 accuracy / 0.9358 F1, and the
representation ceiling (per-vector majority) is 0.9055. Accuracy and F1 are
therefore *not* the operating metrics — ranking is: ROC-AUC 0.85–0.88 vs 0.5
for a constant, PR-AUC 0.974–0.981 vs 0.879 prevalence, and FPR/FNR express
the actual product trade-off. Full reasoning:
[`docs/METHODOLOGY.md` §7](../docs/METHODOLOGY.md).

## Serving guidance

| Choose | When |
|---|---|
| `random_forest.joblib` (**default**) | best accuracy/recall on the primary protocol |
| `xgboost.joblib` | 7.5× lower latency (14.5 ms) at −2.2 accuracy points |
| `lightgbm.joblib` | 19× lower latency (5.7 ms), 8× smaller file |
| `logistic_regression.joblib` | lowest FPR (7.5%), best ROC-AUC, 1.4 KB artifact |

Switch without redeploying code: `MODEL_NAME=xgboost.joblib python app.py`
(bare filename inside `models/` only — paths are rejected at boot).
Latencies are single-row `predict()` on the training machine and are meant for
comparison, not as absolute guarantees.

## Environment

Trained with Python 3.14.3, scikit-learn 1.9.0, XGBoost 3.4.1, LightGBM
4.7.0, pandas 3.0.5, numpy 2.5.2 (`trained_at_utc` recorded in
`registry.json`). Fits are single-threaded with explicit `random_state=42`, so
re-running the benchmark reproduces every metric within `1e-6`.

## Known limitations

1. **No probability calibration** — `score` is raw `predict_proba`; the
   threshold is fixed at 0.5.
2. **Prevalence shift** — the pool is 87.5% phishing; precision will fall at
   real-world base rates. Recall and FPR transfer better than precision.
3. **Provenance unverified** — dataset collection dates and licence unknown.
4. **No dataset shift monitoring** — metrics are frozen at training time.
5. **Unsigned artifacts** — integrity relies on SHA-256 manifests plus code
   review (see [`docs/threat-model.md` T14](../docs/threat-model.md)).

## Retraining and verification

```bash
python train_model.py                    # refit all experiments, refresh artifacts
python train_model.py --check            # refit and assert metrics reproduce
python evaluate_models.py --check        # rescore shipped artifacts vs metrics.json
pytest -m "not network"                  # 180+ tests
```

## History (kept for traceability)

* **Legacy baseline** artifacts quoted 85.95% / 85.70% accuracy from a single
  random split over 771 unique vectors — leaked by construction; re-measured
  here as worth 8.4–12.2 accuracy points. Withdrawn.
* **Naming incident:** upstream shipped `xg_boost.pkl` whose bytes were
  identical to `random_forest_model.pkl` (a Random Forest under two names).
  Both ambiguous files were replaced by explicit names and verified for
  prediction parity before removal. Machine-readable provenance was added in
  `registry.json`, with `train_model.py` now the only writer.
* **Page/diagnostic features** — the `full_legacy16` feature set is retained
  in the benchmark for comparison only; it is not servable because of
  `Web_Traffic`.
