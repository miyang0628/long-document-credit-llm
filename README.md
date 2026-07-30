# Simplicity Wins in Long-Document Credit Analysis

Code and experiments for the paper *"Text-Disclosure Credit: Long-Document
Processing Strategies for LLM-Based Bankruptcy Prediction"* (under review;
author information withheld for anonymous review).

## Overview

10-K filings run 10,000–30,000 words — far beyond a single LLM context. Which
strategy best extracts a bankruptcy signal from such long disclosures? Testing
four strategies across three model tiers, we find:

- **Elaborate strategies do not beat plain truncation.** Chunking, summarization,
  and keyword extraction are statistically indistinguishable from simply feeding
  the first 8K tokens (bootstrap CIs overlap).
- **They just cost more.** Chunking costs ~4× truncation and summarization ~5×,
  for no gain in ROC-AUC.
- **Model strength — not strategy — drives performance** (ROC-AUC rises from
  0.83 to 0.91 across tiers).
- **A strong model reading text alone beats the numeric-financials baseline**
  (0.910 vs 0.862), though weaker models only match it.

In short: for long-document credit analysis, a simple truncation plus a strong
model is the efficient choice.

## Data

- **Text-based bankruptcy dataset** — 10-K MD&A + Risk Factors (full text),
  39 numeric financial variables, and a bankruptcy label. Mendeley Data,
  DOI 10.17632/stf3kg7fw3.
- 222 firms (111 train / 111 test), balanced (~50% bankrupt), case-control
  matched. MD&A median ~10K words (p90 ~30K).

The raw dataset is **not redistributed** here; `00_data_acquisition` downloads it.

## Repository structure

.
├── notebooks/
│ ├── 00_data_acquisition.ipynb # download & validate public data
│ ├── 01_eda.ipynb # size, imbalance, text length, confound
│ ├── 02_baseline.ipynb # numeric / length / combined baselines
│ ├── 03_longtext_strategies.ipynb # pilot + full run (LLM calls)
│ └── 04_results_viz.ipynb # figures
├── requirements.txt
├── LICENSE
└── README.md


Artifacts (tables, figures, cached LLM outputs) are written to `artifacts/` at
run time and are not committed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes tiktoken for token counting
```

The strategy experiments (`03_longtext_strategies`) call the OpenAI API. Provide
a key via a git-ignored `.env`:

OPENAI_API_KEY=sk-...


## Reproducing the results

Run in order: `00 → 01 → 02 → 03 → 04`.

- `01_eda`, `02_baseline` are deterministic (scikit-learn, no API).
- `03_longtext_strategies` runs a 50-firm pilot (all 4 strategies, to measure
  cost) then the full 222-firm run (3 strategies; summarization is dropped after
  the pilot shows it matches truncation at ~5× cost). Outputs cache to
  `artifacts/inference/*.jsonl`; re-running does not re-incur API cost.
- `04_results_viz` assembles the figures.

Strategies compared: **truncation** (first 8K tokens), **extraction**
(risk-keyword sentences), **chunking** (8K chunks, judged and aggregated), and
**summarization** (pilot only). Numeric and text-length baselines are established
in `02` via repeated stratified cross-validation.

## Key results

| Strategy (gpt-5.4) | ROC-AUC [95% CI] | Cost (222 firms) |
|--------------------|------------------|------------------|
| truncation         | 0.910 [0.870, 0.944] | $4.2 |
| chunking           | 0.898 [0.852, 0.938] | $17.0 |
| extraction         | 0.893 [0.845, 0.932] | $2.0 |
| numeric baseline   | 0.862 ± 0.050    | — |

Strategy CIs overlap (no significant difference); performance rises with model
strength (0.83 → 0.91 across tiers). See `artifacts/` after running.

## License

Released under the MIT License. See `LICENSE`.

The dataset is licensed separately by its authors and is not included here.

## Citation

Citation details withheld during anonymous review.
