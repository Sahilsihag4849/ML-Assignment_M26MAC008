# ML Assignment 2

- **Name:** Sahil Sihag
- **Roll number:** M26MAC008
- **Seed:** 26008 
- **Repository:** https://github.com/Sahilsihag4849/ML-Assignment_M26MAC008

## Environment
Python <output of `python --version`>. Main packages: numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, joblib 1.6.0 (all pinned in `requirements.txt`).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to reproduce
Run from the project root, in this order:

```
python src/make_split.py            # already done; data/split.npz is committed. Do not rerun.
python src/q1a_tree.py              # single tree grid, freezes the tree setting
python src/q1b_bagging.py           # bagging with B up to 200
python src/q1c_diagnostics.py       # OOB accuracy, disagreement, vote margin, chooses B
python src/q1d_compare.py           # tree vs bagging, clean and masked test sets
python src/q2a_toy.py               # one logged SAMME round on a 9-sample toy problem
python src/q2b_boosting_grid.py     # boosting grid, chooses depth and rounds
python src/q2c_compare.py           # boosting vs bagging, heavily weighted face
python src/q2d_masked_compare.py    # all three models, clean and masked
python src/q3_refits.py             # 30 resampled refits; about 30 minutes on 15 cores
python src/q3_analysis.py           # bias squared, variance, final table
```

Each script writes its log to `logs/` and its settings, predictions or figures to `configs/`, `results/` or `figures/`. Scripts later in the list read what earlier ones saved.

## Repository layout
- `src/`: all code. `common.py` holds shared helpers (data loading, mask, vote counting). `samme.py` holds the SAMME update used by both the toy example and the face data.
- `data/split.npz`: the saved 6/2/2 per-identity split (seed 26008) used by every script.
- `configs/`: frozen settings written by the scripts (`frozen_tree.json`, `final_bagging.json`, `final_boosting.json`).
- `logs/`: final logs, one per script.
- `results/`, `models/`: saved predictions, boosting weights and fitted models. `results/q3/` holds the 30 refits.
- `figures/`: plots used in the report.
- `report/`: LaTeX source and PDF.
