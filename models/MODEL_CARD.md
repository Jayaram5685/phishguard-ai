# Model Card — PhishGuard AI (legacy baseline artifacts)

> **Status: LEGACY BASELINE.** These artifacts were inherited from the upstream
> repository and are kept only so the existing API keeps working while the
> pipeline is rebuilt (milestone **P2**). **Do not quote their scores as
> product performance** — see [Known integrity issues](#known-integrity-issues).

## Artifacts

| File | Algorithm | Estimators | SHA-256 (artifact) | Size |
|---|---|---|---|---|
| `models/random_forest.joblib` | `RandomForestClassifier` (scikit-learn) | 500 | `e8dd5405e647ed03fc0538327dbe4f5a65ffea228e2eb9c069165a0dca7083b7` | 17.35 MB |
| `models/xgboost.joblib` | `XGBClassifier` (XGBoost) | 500 | `67017f56dce4bb38cd9d4ae03bf9221e970d55a80d2d7dcb95206a0674737ce8` | 1.36 MB |

Machine-readable provenance lives in [`models/registry.json`](registry.json).

### Naming history (why this migration was necessary)

Upstream shipped three pickle files. Byte-level inspection showed:

| Upstream file | Actual contents |
|---|---|
| `xg_boost.pkl` | `RandomForestClassifier` — **identical byte-for-byte to `random_forest_model.pkl`** (SHA-256 `6494ecfa…23fd`) |
| `random_forest_model.pkl` | `RandomForestClassifier` (same bytes) |
| `xgboost_model.pkl` | `XGBClassifier` (the real XGBoost model) |

`train_model.py` wrote the Random Forest into **two** filenames, one of which was
named `xg_boost.pkl`. Both ambiguous files were replaced by explicit names and
verified for **prediction parity** (identical `predict` and `predict_proba` over
500 unique dataset rows) before removal.

## Training data

- **Dataset:** `DataFiles/5.urldata.csv` — 10,000 rows, balanced (5,000 / 5,000).
- **Provenance:** assembled upstream from Kaggle-style phishing collections;
  legitimate URLs attributed to Alexa top-site lists, phishing URLs attributed
  to PhishTank feeds (`DataFiles/1.Benign_list_big_final.csv`,
  `DataFiles/2.online-valid.csv`). Exact collection dates and licenses were
  **not documented upstream** → recorded as `TODO / UNVERIFIED`.
- **Split (upstream procedure):** random 80/20, `random_state=42`, no
  stratification check, no domain grouping.

## Feature schema (input contract)

Models expect exactly these 16 columns, in this order, as numeric values:

```
Have_IP, Have_At, URL_Length, URL_Depth, Redirection, https_Domain,
TinyURL, Prefix/Suffix, DNS_Record, Web_Traffic, Domain_Age, Domain_End,
iFrame, Mouse_Over, Right_Click, Web_Forwards
```

Binary columns are `0/1` except `URL_Depth` (count). `label_semantics`: `0 =
legitimate`, `1 = phishing`.

## Compatibility

- Trained with **scikit-learn 1.7.2** (evidence: sklearn's
  `InconsistentVersionWarning` declares `1.7.2` in the pickle state).
  `requirements.txt` therefore floors scikit-learn at `1.7.2`.
- XGBoost training version is **not recoverable** from the artifact → recorded
  as `unknown` rather than guessed.

## Known integrity issues

1. **Severe train/test leakage.** The 10,000 rows contain only **771 unique
   feature vectors** and **3,553 unique domains**. A random split therefore
   places near-identical rows on both sides of the split, so any accuracy
   reported from it measures memorisation, not generalisation to unseen
   phishing domains.
2. **Obsolete feature.** `Web_Traffic` was derived from the Alexa Rank API
   (`data.alexa.com`), which Alexa Internet discontinued in 2022. At inference
   time the call always fails and the feature degenerates to a constant.
3. **Broken feature at inference.** `Have_IP` was computed against the *whole
   URL string* rather than the hostname, so it is effectively always `0` in
   serving, although it carried signal in training data.
4. **No probability calibration.** Both models emit raw class labels only.

## Metrics — provenance, not claims

| Source | Value | Caveat |
|---|---|---|
| upstream `README.md` | RF 85.95%, XGBoost 85.70% accuracy | single random split on leaked data; no script in repo reproduces it |
| upstream training notebook | RF 82.4%, XGBoost 85.8% test accuracy | same leaked split |

These numbers are recorded for **traceability only**. They are marked
`UNVERIFIED` and must not be used in the README, dashboard, or research paper
as evidence of detection quality. Reproducible metrics replace them in
milestone **P8** (evaluation framework).

## Retraining

```bash
python train_model.py     # writes models/*.joblib + registry entries (P2 rewrite)
```

The P2 rewrite adds: deduplicated/domain-grouped splits, a domain-disjoint
evaluation protocol, removal of the dead Alexa feature, and explicit model
versioning.
