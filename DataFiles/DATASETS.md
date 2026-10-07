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
- **Unique feature vectors (features only):** **771** → 9,229 rows are duplicates
  of some other row's feature vector

### Consequence (why the old metrics were withdrawn)

Because only 771 distinct feature vectors exist, the upstream procedure
(`train_test_split(test_size=0.2, random_state=42)`) places near-identical rows
in both train and test. Reported accuracies from that split measure
memorisation, not generalisation, and are therefore recorded in
[`models/MODEL_CARD.md`](../models/MODEL_CARD.md) as `UNVERIFIED`.

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
