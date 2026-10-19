# Dataset Card — PhishGuard AI

> Every figure below was measured from the files in this directory.
> Anything that could not be verified from the repository itself is marked
> **TODO / UNVERIFIED** rather than estimated.

## Files

| File | Rows (incl. header) | Columns | Role |
|---|---|---|---|
| `1.Benign_list_big_final.csv` | 35,378 | 1 (URL) | raw legitimate URL list |
| `2.online-valid.csv` | 14,859 | 8 (PhishTank schema) | raw phishing URL list (PhishTank) |
| `3.legitimate.csv` | 5,001 | 18 | pre-extracted features, label `0` |
| `4.phishing.csv` | 5,001 | 18 | pre-extracted features, label `1` |
| `5.urldata.csv` | 10,001 | 18 | combined training set used by `train_model.py` |

## Training set — measured facts

Measured with `pandas` over `5.urldata.csv`:

- **Shape:** 10,000 rows × 18 columns (`Domain`, 16 features, `Label`)
- **Class distribution:** 5,000 legitimate (`0`) / 5,000 phishing (`1`) — balanced
- **Unique domains:** 3,553 → 6,447 rows share a domain with another row
- **Unique feature vectors (legacy 16 features):** **771** → 9,229 rows are
  duplicates of some other row's feature vector
- **Unique feature vectors (features-v2, 15 features):** **581** — dropping the
  dead `Web_Traffic` column merges further rows together. This is the count
  that matters for training: the model sees at most 581 distinct inputs.
- **Feature vectors carrying both labels (features-v2):** 94, covering 6,593
  rows — the representation cannot separate them (information ceiling, not a
  modelling failure)

### Consequence (why the old metrics were withdrawn)

Because only 771 distinct feature vectors (581 under features-v2) exist for
10,000 rows, the upstream procedure
(`train_test_split(test_size=0.2, random_state=42)`) places near-identical rows
in both train and test. Reported accuracies from that split measure
memorisation, not generalisation: re-measured here, the leak is worth
**8.4–12.2 accuracy points** depending on the model. The legacy figures are
recorded in [`models/MODEL_CARD.md`](../models/MODEL_CARD.md) for traceability
only, and the corrected protocol is documented in
[`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md).

### Schema inconsistencies found

- `4.phishing.csv` names the column `Tiny_URL`; every other file uses `TinyURL`.
- `1.Benign_list_big_final.csv` and `2.online-valid.csv` are raw URL lists with
  **no header row semantics** — the first data row is read as a header by naive
  `read_csv` calls.

## Provenance & licensing

| Field | Value |
|---|---|
| Collection dates | **TODO / UNVERIFIED** — not recorded upstream |
| Legitimate URL source | attributed upstream to Alexa top-site lists (service discontinued 2022) |
| Phishing URL source | PhishTank (`2.online-valid.csv` carries `phish_id`, `submission_time` columns) |
| Original aggregator | Kaggle-style phishing-detection dataset — exact URL **TODO / UNVERIFIED** |
| License | **TODO / UNVERIFIED** — no license file accompanied the data upstream |

Raw CSVs are committed because they were already public in the upstream
repository and are required to reproduce the legacy baseline. Before any
publication or redistribution, licensing must be confirmed — this is tracked as
an open item in the research documentation.

## Intended use

- Training and evaluating phishing-URL classifiers **for research**.
- Demonstrating leakage-aware evaluation methodology.

## Out-of-scope use

- Claiming real-world detection rates from this data alone.
- Production block-listing decisions without the live intelligence layers
  (DNS/RDAP/certificate/brand) described in the architecture.
