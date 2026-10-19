# Evaluation Methodology — PhishGuard AI

> Every number in this document was measured from the files committed to this
> repository. Nothing is quoted from upstream, and nothing is estimated.
> Reproduce any table below with the commands in
> [Reproducibility](#8-reproducibility).

## 1. Why this document exists

The upstream project reported **85.95% accuracy** from a single random split.
That number is not wrong arithmetic — it is an *uninformative* measurement: the
split it came from leaks. This document records the protocol that replaced it,
so a reviewer can judge the shipped models without trusting anyone's prose.

Three principles govern everything below:

1. **No leakage across the train/test boundary** unless a protocol says so
   explicitly (and labels itself a baseline).
2. **No metric without its denominator** — every proportion is reported with
   its sample size and, where it matters, a 95% interval.
3. **No claim that the code does not enforce** — invariants asserted here are
   asserted again by the test suite.

## 2. The dataset

`DataFiles/5.urldata.csv`, SHA-256
`5c74254ce576a730beaea2ed15c3acbfd7e02dad853b552d248b5af5a034b188`
(10,000 rows × 18 columns = `Domain` + 16 legacy features + `Label`).

| Measured fact | Value |
|---|---|
| Rows / labels | 10,000 — 5,000 `0` (legitimate), 5,000 `1` (phishing) |
| Nulls | 0 |
| Unique domains | 3,553 |
| Unique feature vectors, legacy 16 columns | **771** |
| Unique feature vectors, features-v2 (15 columns) | **581** |
| Vectors carrying both labels (features-v2) | 94, covering **6,593 rows** |
| Domains carrying both labels | 3 (8 rows) |
| Rows after deduplication | 4,374 — 529 `0`, 3,845 `1` |
| **Training pool** (dedup + mixed domains dropped) | **4,366** — 524 `0`, 3,842 `1` |

Full provenance, licensing caveats and the raw file inventory live in
[`DataFiles/DATASETS.md`](../DataFiles/DATASETS.md).

### 2.1 The three distortions in the source data

* **Duplication.** 10,000 rows collapse to 4,374 unique rows: 47% of the file
  is a repeat. Any row-level split therefore puts copies of the *same* row on
  both sides.
* **Feature collisions.** Only 581 distinct feature vectors exist for 10,000
  rows, and 94 of them carry **both** labels (6,593 rows). The 15-feature
  representation cannot separate those rows — this is an information ceiling,
  not a modelling failure.
* **Prevalence inversion.** After deduplication the pool is 87.5% phishing
  (3,842/4,366), even though the raw file is balanced. Deduplication removes
  *legitimate* rows far more aggressively (5,000 → 529) because the legitimate
  source list is heavily repeated. Every downstream metric must be read with
  this in mind (§7).

## 3. Preprocessing (deterministic, no fitted state)

```
load_raw()  →  schema-validate against features-v2
            →  deduplicate()            10,000 → 4,374 rows
            →  drop_mixed_domains()     4,374 → 4,366 rows (3 domains, 8 rows)
            →  training pool
```

* **No target encoding, no imputation, no fitted transformers** — preprocessing
  is pure and can never learn from the test split.
* **Row ids are preserved.** Splits keep each row's original index, so a test
  prediction traces back to a line in the CSV, and "train ∪ test = pool with
  disjoint row ids" is verified as set equality rather than a row count
  (`phishguard/ml/split.py::_assert_exact_partition`).
* Mixed-label domains are *refused*, not silently split: a domain whose rows
  carry both labels makes a domain-disjoint split ill-defined
  (`SplitError`, covered by `tests/test_split.py`).

## 4. Split protocols

All protocols take `test_fraction = 0.2` and `seed = 42`, and are deterministic
(verified by re-running and by `tests/test_split.py`).

| Protocol | Source | Train | Test | Overlap | Purpose |
|---|---|---|---|---|---|
| `random` | raw 10,000 | 8,000 | 2,000 (1,012 `0` / 988 `1`) | **leaked by design** | reproduces the legacy procedure as a baseline |
| `domain_disjoint` | pool 4,366 | 3,488 (2,839 domains) | 878 (711 domains; 106 `0` / 772 `1`) | **0 domains** | **primary protocol** — generalisation to unseen domains |
| `vector_disjoint` | pool 4,366 | 3,223 (464 vectors) | 1,143 (116 vectors; 120 `0` / 1,023 `1`) | **0 vectors** | stress test — feature combinations never seen in training |

Why `domain_disjoint` is primary: phishing kits cluster by domain, and the raw
file contains only 3,553 domains for 10,000 rows. A row-level split lets the
model score a *known* domain on the test side, which measures memorisation.
`domain_disjoint` stratifies domains by label, then withholds whole domains.

Why `vector_disjoint` complements it: domains being disjoint does **not**
imply features are — 205 distinct feature vectors appear on both sides of the
`domain_disjoint` split. `vector_disjoint` removes that shortcut entirely.

## 5. Features — features-v2 (15 columns)

Canonical order and semantics are declared once in
[`phishguard/features.py`](../phishguard/features.py) and enforced for
training, evaluation, extraction and the API. Value ranges are validated; a
mapping that violates the schema raises `SchemaError` and never reaches a
model.

| Group | Features |
|---|---|
| url (8) | `Have_IP`, `Have_At`, `URL_Length`, `URL_Depth`, `Redirection`, `https_Domain`, `TinyURL`, `Prefix/Suffix` |
| domain (3) | `DNS_Record`, `Domain_Age`, `Domain_End` |
| page (4) | `iFrame`, `Mouse_Over`, `Right_Click`, `Web_Forwards` |

### 5.1 What changed from the legacy 16-column contract

* **`Have_IP` is now hostname-scoped.** The legacy implementation called
  `ipaddress.ip_address()` on the *whole URL string*, which can never succeed
  when a scheme is present — the feature was effectively dead in serving.
  Reconstructing the dataset from the raw source lists shows the hostname
  rule matches **4,964/4,964** reconstructed rows, while the whole-URL
  variant is contradicted by 37 rows (IPs embedded in paths, labelled
  legitimate).
* **`Web_Traffic` was removed.** Its source (Alexa rank) was discontinued in
  2022, so at serving time the value can only be a constant. See §6.2 for the
  measured cost of that mismatch.

### 5.2 Ablation ladder

Each level is a strict prefix of the next, so the ladder isolates the marginal
value of each feature group (`phishguard.features.ABLATIONS`):

`url_only` (8) ⊂ `url_domain` (11) ⊂ `url_domain_page` (15, shipped) ⊂
`full_legacy16` (16, diagnostic only).

## 6. Results

Unless stated otherwise: primary protocol `domain_disjoint`, shipped feature
set `url_domain_page`, seed 42, n = 878 test rows, single-threaded fits.

### 6.1 Primary protocol

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | Bal. acc.* | p95 latency | Size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `logistic_regression` | 0.6948 | 0.9846 | 0.6632 | 0.7926 | **0.8753** | **0.9814** | **0.0755** | 0.3368 | 0.7939 | 6.7 ms | 1.4 KB |
| `random_forest` | **0.7472** | 0.9725 | **0.7332** | **0.8360** | 0.8496 | 0.9744 | 0.1509 | 0.2668 | 0.7911 | 108.3 ms | 12.8 MiB |
| `xgboost` | 0.7255 | 0.9801 | 0.7021 | 0.8181 | 0.8632 | 0.9796 | 0.1038 | 0.2979 | **0.7991** | 14.5 ms | 911 KiB |
| `lightgbm` | 0.7244 | 0.9749 | 0.7047 | 0.8180 | 0.8598 | 0.9790 | 0.1321 | 0.2953 | 0.7863 | 5.7 ms | 1.6 MiB |

\* Balanced accuracy = (recall + 1 − FPR) / 2 — derived from the committed
metrics, not a separate model run.

Uncertainty is reported by the benchmark itself: every metric carries a 95%
Wilson interval in `models/metrics.json`. For `random_forest`, accuracy
0.7472 has CI **[0.7174, 0.7748]** and FPR 0.1509 has CI
**[0.0951, 0.2312]** — with 106 legitimate test rows, the FPR estimate is
necessarily wide, and should be read as such.

Latency is single-row `predict()` on the recording machine; it is recorded to
compare models against each other, not to advertise absolute speed.

### 6.2 What `Web_Traffic` would have cost (measured, not asserted)

Training on the legacy 16 columns includes `Web_Traffic`, but serving cannot
supply it (Alexa is dead). Measured on the same split — train on 16 columns,
then evaluate once as-is and once with `Web_Traffic` forced to the only value
serving can produce (`1`):

| Model | 16 features | serving forces `1` | cost | features-v2 (15, shipped) |
|---|---|---|---|---|
| `logistic_regression` | 0.7380 | 0.6936 | **−4.44 pts** | 0.6948 |
| `random_forest` | 0.7722 | 0.7335 | **−3.87 pts** | 0.7472 |
| `xgboost` | 0.7403 | 0.7027 | **−3.76 pts** | 0.7255 |
| `lightgbm` | 0.7528 | 0.7141 | **−3.87 pts** | 0.7244 |

Dropping the column beats serving a constant for all four models, which is why
the dead feature is excluded from the contract rather than defaulted. (The
column is also weak: it is `1` for 84.6% of rows and correlates with the
*legitimate* class — mean 0.885 legitimate vs 0.806 phishing.)

### 6.3 Protocol comparison — what leakage was worth

| Model | `random` (leaked) | `domain_disjoint` (primary) | `vector_disjoint` | inflation |
|---|---|---|---|---|
| `logistic_regression` | 0.7790 / 0.8529 | 0.6948 / 0.8753 | 0.7743 / 0.9050 | **+8.42 pts** |
| `random_forest` | 0.8490 / 0.9016 | 0.7472 / 0.8496 | 0.8171 / 0.9052 | **+10.18 pts** |
| `xgboost` | 0.8475 / 0.9020 | 0.7255 / 0.8632 | 0.8066 / 0.9021 | **+12.20 pts** |
| `lightgbm` | 0.8450 / 0.8995 | 0.7244 / 0.8598 | 0.8548 / 0.9022 | **+12.06 pts** |

*(accuracy / ROC-AUC; inflation = `random` − `domain_disjoint` accuracy.)*

The leaked protocol is worth 8.4–12.2 accuracy points — that gap *is* the
leakage. It also explains upstream's 85.95% claim: our `random` protocol
reproduces 84.9% for the Random Forest from first principles, so upstream's
number was an honest reading of an instrument that could not measure
generalisation.

### 6.4 Ablation ladder (accuracy / ROC-AUC, `domain_disjoint`)

| Model | `url_only` (8) | `url_domain` (11) | `url_domain_page` (15) | `full_legacy16` (16)† |
|---|---|---|---|---|
| `logistic_regression` | 0.6765 / 0.8424 | 0.6800 / 0.8672 | 0.6948 / 0.8753 | 0.7380 / 0.8811 |
| `random_forest` | 0.6845 / 0.8366 | 0.7005 / 0.8446 | 0.7472 / 0.8496 | 0.7722 / 0.8563 |
| `xgboost` | 0.6754 / 0.8388 | 0.6777 / 0.8591 | 0.7255 / 0.8632 | 0.7403 / 0.8734 |
| `lightgbm` | 0.6743 / 0.8382 | 0.6880 / 0.8560 | 0.7244 / 0.8600 | 0.7528 / 0.8678 |

† Diagnostic only — not servable (§6.2).

Page features earn their place: moving `url_domain` → `url_domain_page` is the
largest single jump in the ladder (e.g. Random Forest 0.7005 → 0.7472).
URL-only features alone are close to useless for classification, which is why
the extraction layer still performs (SSRF-guarded) page and WHOIS lookups.

## 7. Reading these numbers honestly

**The headline metrics on this split are traps.** The test split is 87.9%
phishing (772/878). Two baselines make that concrete:

| Baseline (no model, test split n=878) | Accuracy | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|
| Always predict "phishing" | **0.8793** | **0.9358** | 0.5000 | 0.8793 | 1.0000 |
| Best possible given the 15 features (per-vector majority) | **0.9055** | — | — | — | — |
| Shipped models (§6.1) | 0.6948 – 0.7472 | 0.7926 – 0.8360 | 0.8496 – 0.8753 | 0.9744 – 0.9814 | 0.0755 – 0.1509 |

Therefore:

* **Accuracy and F1 are not the operating metrics here.** A classifier that
  blocks everything scores 0.8793/0.9358 while being worthless.
* **Ranking is where the models earn their keep:** ROC-AUC 0.85–0.88 against
  0.5 for a constant, and PR-AUC 0.974–0.981 against 0.879 prevalence.
* **FPR/FNR express the product trade-off:** LogisticRegression flags only
  7.5% of legitimate sites (FPR 0.0755) at 66% phishing recall; RandomForest
  catches 73% of phishing at the cost of 15% legitimate sites.
* **The representation ceiling is ~0.905**, so no retraining on these 15
  features can reach the 0.95+ accuracies often advertised for this task.
* **Intervals matter:** with 106 legitimate test rows, FPR's 95% CI spans
  0.095–0.231. Do not quote a point FPR without its interval.

## 8. Reproducibility

```bash
pip install -r requirements-dev.txt

pytest -m "not network"        # 180 offline tests: schema, splits, registry, API
ruff check .                   # lint

python evaluate_models.py --check   # load each shipped artifact, rescore the
                                    # split, compare with models/metrics.json
python train_model.py --check       # refit all 24 experiments, compare metrics
```

Both `--check` commands exit non-zero on drift. Locally the default tolerance
is `1e-6`; CI (Linux/3.12, a different platform from the training machine)
runs the artifact-parity gate at `5e-3` and the refit gate at `2e-2` — see
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

Determinism comes from: an explicit `random_state=42` on every estimator;
single-threaded fits (`n_jobs=1`, `nthread=1`, LightGBM
`deterministic=True, force_row_wise=True`); a content-addressed dataset
(SHA-256 recorded in `models/metrics.json`); and no fitted preprocessing.
Verified locally: `train_model.py --check` reproduces all 24 experiments to
within `1e-6`, including metrics, on repeat runs.

One honest caveat, measured in CI: **XGBoost's `hist` tree builder is not
bit-reproducible across platforms.** Refitting on Linux/Python 3.12 against
metrics recorded on Windows/Python 3.14 drifted by up to `1.04e-2` (a single
ablation, `xgboost/domain_disjoint/url_domain`); every other experiment,
including the shipped artifacts, stayed inside `5e-3`. The CI refit gate is
therefore set to `2e-2` — well above float noise, far below the 8+ point moves
a wrong split, lost seed or changed feature set produces.

`models/metrics.json` (benchmark) and `models/registry.json` (shipped
artifacts) are cross-checked by `tests/test_ml_pipeline.py` and
`tests/test_model_registry.py`: every registry entry must point at metrics
that exist in the benchmark, and at bytes that match its SHA-256.

## 9. Limitations and threats to validity

1. **Provenance is unverified.** Collection dates and the licence of the
   source lists were never recorded upstream (`DataFiles/DATASETS.md`). The
   data supports method development, not published detection-rate claims.
2. **Single dataset, single seed.** Seed 42 only; the protocols are
   deterministic but not yet bootstrapped across resamples. Wilson intervals
   quantify test-set sampling, not dataset shift.
3. **Prevalence shift.** The pool is 87.5% phishing while the live web is not.
   Precision/PR-AUC will fall in production at any lower phishing prevalence
   (base-rate effect); recall and FPR transfer better than precision.
4. **Offline reconstruction ≠ live extraction.** Training-time page/WHOIS
   signals were reconstructed from committed data; live lookups fail or
   degrade routinely, and the extractor's "no intelligence" values (§5) shift
   the input distribution. The API reports exactly which signals were
   unavailable (§ `phishguard.explain`).
5. **No probability calibration.** `score` is the model's raw
   `predict_proba` output, not a calibrated risk. Thresholds are fixed at 0.5.
6. **DNS rebinding / resolver latency** in the SSRF guard are residual risks
   with explicit mitigations — see [`docs/threat-model.md`](threat-model.md).

## 10. Where each claim is enforced

| Claim | Enforced by |
|---|---|
| Splits are an exact, disjoint partition of the pool | `phishguard/ml/split.py::_assert_exact_partition`, `tests/test_split.py` |
| Domain split has zero domain overlap; vector split zero vector overlap | `tests/test_split.py` |
| Splits are deterministic for a seed | `tests/test_split.py` |
| Every artifact matches its registry checksum and metrics | `tests/test_model_registry.py`, `tests/test_ml_pipeline.py` |
| Features-v2 is the only schema (API, training, extraction) | `tests/test_features.py`, `tests/test_data.py`, app import-time assertion |
| Committed metrics reproduce from scratch | `train_model.py --check` (CI) |
| Shipped artifacts reproduce committed metrics | `evaluate_models.py --check` (CI) |
| Explanations match the scored feature vector | `tests/test_explain.py` |
