import os
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from common import load_data, mask_faces, project_folder
from samme import fit_samme, samme_scores

N_REPS = 30                                   # test with 2 first, then set back to 30
N_JOBS = max(1, (os.cpu_count() or 2) - 1)    # repetitions run in parallel
K = 40

(Xtr, ytr, _, _, Xte, yte), seed = load_data()
Xte_masked = mask_faces(Xte)

tree_cfg = json.load(open(os.path.join(project_folder, "configs", "frozen_tree.json")))
B_final = json.load(open(os.path.join(project_folder, "configs", "final_bagging.json")))["B"]
boost_cfg = json.load(open(os.path.join(project_folder, "configs", "final_boosting.json")))
depth, R = boost_cfg["depth"], boost_cfg["rounds"]
assert B_final >= 100

# the three tree settings used in Q3(b)
CONFIGS = {"shallow": dict(max_depth=8, min_samples_leaf=tree_cfg["min_samples_leaf"]),
           "selected": tree_cfg,
           "deep": dict(max_depth=None, min_samples_leaf=1)}
out_dir = os.path.join(project_folder, "results", "q3")


def resample_training_set(r):
    """Within each person, draw 6 faces WITH replacement from that person's 6 training faces."""
    rng = np.random.default_rng([seed, 3, r])
    chosen = []
    for person in range(K):
        own = np.where(ytr == person)[0]
        chosen.extend(rng.choice(own, size=6, replace=True))
    return np.array(chosen)


def one_hot(pred):
    v = np.zeros((len(pred), K))
    v[np.arange(len(pred)), pred] = 1
    return v


def bagging(Xf, yf, cfg, n_trees, rng, seed_base):
    """Returns vote counts on the test set for the first 100 trees and for all n_trees."""
    clean = np.zeros((len(Xte), K))
    masked = np.zeros((len(Xte), K))
    rows = np.arange(len(Xte))
    for b in range(n_trees):
        s = rng.integers(0, len(yf), size=len(yf))                  # bootstrap sample
        tree = DecisionTreeClassifier(**cfg, random_state=seed_base + b).fit(Xf[s], yf[s])
        clean[rows, tree.predict(Xte)] += 1
        masked[rows, tree.predict(Xte_masked)] += 1
        if b == 99:
            first100 = (clean.copy(), masked.copy())
    return first100, (clean, masked)


def run_one_repetition(r):
    path = os.path.join(out_dir, f"rep_{r:02d}.npz")
    if os.path.exists(path):                                        # already done, skip
        return
    idx = resample_training_set(r)
    Xf, yf = Xtr[idx], ytr[idx]
    out = {}

    # single trees (class vector = one-hot of the predicted class)
    for name, cfg in CONFIGS.items():
        tree = DecisionTreeClassifier(**cfg, random_state=seed + r).fit(Xf, yf)
        out[f"tree_{name}_clean"] = one_hot(tree.predict(Xte))
        out[f"tree_{name}_masked"] = one_hot(tree.predict(Xte_masked))

    # bagging: 100 trees for each setting; the selected setting runs B_final trees
    for tag, (name, cfg) in enumerate(CONFIGS.items()):
        n_trees = B_final if name == "selected" else 100
        rng = np.random.default_rng([seed, 4, r, tag])
        first100, final = bagging(Xf, yf, cfg, n_trees, rng, seed_base=1000 * r)
        out[f"bag100_{name}_clean"], out[f"bag100_{name}_masked"] = first100
        if name == "selected":
            out["bag_final_clean"], out["bag_final_masked"] = final

    # boosting (class vector = normalised weighted vote q)
    learners, alphas, _ = fit_samme(Xf, yf, K, R, depth, seed + r)
    total = max(alphas.sum(), 1e-12)
    out["boost_clean"] = samme_scores(learners, alphas, Xte, K) / total
    out["boost_masked"] = samme_scores(learners, alphas, Xte_masked, K) / total

    np.savez_compressed(path, **out)


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    Parallel(n_jobs=N_JOBS, verbose=10)(delayed(run_one_repetition)(r) for r in range(N_REPS))
    print("all repetitions done")
