import os
import json
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_data, mask_faces, project_folder, vote_counts, disagreement
from samme import samme_scores

(Xtr, ytr, _, _, Xte, yte), seed = load_data()
K = 40

# ---------- load the three frozen models ----------
tree_cfg = json.load(open(os.path.join(project_folder, "configs", "frozen_tree.json")))
B = json.load(open(os.path.join(project_folder, "configs", "final_bagging.json")))["B"]
boost_cfg = json.load(open(os.path.join(project_folder, "configs", "final_boosting.json")))
depth, R = boost_cfg["depth"], boost_cfg["rounds"]

tree = DecisionTreeClassifier(**tree_cfg, random_state=seed).fit(Xtr, ytr)     # same refit as Q1a
bag = np.load(os.path.join(project_folder, "results", "q1_bagging.npz"))       # saved tree predictions
learners, alphas = joblib.load(os.path.join(project_folder, "models", "boosting_all_depths.joblib"))[depth]
learners, alphas = learners[:R], alphas[:R]
W = np.load(os.path.join(project_folder, "results", "q2_boosting.npz"))[f"d{depth}_W"][:R]

Xmasked = mask_faces(Xte)
sets = {"clean": (Xte, bag["pred_test"][:B]), "masked": (Xmasked, bag["pred_test_masked"][:B])}
log, res = [], {}

# ---------- accuracy and mean winning vote/probability ----------
for name, (X, bag_preds) in sets.items():
    counts = vote_counts(bag_preds)
    res[name] = {"tree": tree.predict_proba(X), "bagging": counts / B,
                 "boosting": samme_scores(learners, alphas, X, K) / alphas.sum(), "counts": counts}
    for model in ["tree", "bagging", "boosting"]:
        p = res[name][model]
        log.append(f"{name:6s} {model:8s} acc={(p.argmax(1) == yte).mean():.4f}  mean winning={p.max(1).mean():.4f}")
    log.append(f"{name:6s} mean disagreement among bagged trees = {disagreement(counts, B):.4f}")

# ---------- does the single tree even look at the masked block? ----------
t = tree.tree_
split_nodes = np.where(t.children_left != -1)[0]              # nodes that are splits (not leaves)
pixel = t.feature[split_nodes]                                # pixel index used at each split
row, col = pixel // 64, pixel % 64
in_block = (row >= 24) & (row < 40) & (col >= 24) & (col < 40)
log.append(f"tree splits that use a pixel inside the masked block: {int(in_block.sum())} of {len(split_nodes)}")
for node in split_nodes[in_block]:
    log.append(f"  that split: pixel {t.feature[node]}, threshold {t.threshold[node]:.4f}, "
               f"training samples reaching it: {t.n_node_samples[node]}")
changed_leaf = (tree.apply(Xte) != tree.apply(Xmasked)).sum()
log.append(f"test faces whose leaf changes under the mask: {int(changed_leaf)} of {len(yte)}")

# ---------- faces where bagging and boosting react differently to the mask ----------
ok = {(name, model): res[name][model].argmax(1) == yte for name in sets for model in ["bagging", "boosting"]}
boost_breaks = ok["clean", "bagging"] & ok["masked", "bagging"] & ok["clean", "boosting"] & ~ok["masked", "boosting"]
bag_breaks = ok["clean", "boosting"] & ok["masked", "boosting"] & ok["clean", "bagging"] & ~ok["masked", "bagging"]
log.append(f"bagging stays right, boosting breaks: {int(boost_breaks.sum())} faces; "
           f"boosting stays right, bagging breaks: {int(bag_breaks.sum())} faces")

candidates = np.where(boost_breaks | bag_breaks)[0]
if len(candidates) > 0:
    face = int(candidates[0])                                 # first one by test index
    person = int(yte[face])
    log.append(f"\nCHOSEN FACE: test index {face}, true person {person}")
    for name in sets:
        r = res[name]
        c = r["counts"][face]
        log.append(f"{name:6s} tree says {r['tree'][face].argmax()} | bagging says {r['bagging'][face].argmax()} "
                   f"(true-class votes {int(c[person])}/{B}, top {int(c.max())}/{B}, "
                   f"tree disagreement {1 - (c * (c - 1)).sum() / (B * (B - 1)):.3f}) | "
                   f"boosting says {r['boosting'][face].argmax()} "
                   f"(true-class q {r['boosting'][face, person]:.4f}, top q {r['boosting'][face].max():.4f})")

    before = np.array([l.predict(Xte[face:face + 1])[0] for l in learners])
    after = np.array([l.predict(Xmasked[face:face + 1])[0] for l in learners])
    changed = before != after
    log.append(f"boosting learners whose vote changes with the mask: {int(changed.sum())}/{R}, "
               f"carrying {alphas[changed].sum() / alphas.sum():.3f} of total alpha")
    log.append(f"average sample-weight share of person {person}'s 6 training faces: "
               f"{W[:, ytr == person].sum(axis=1).mean():.3f} (uniform would be {6 / len(ytr):.3f})")

print("\n".join(log))
open(os.path.join(project_folder, "logs", "q2d_masked_compare.log"), "w").write("\n".join(log) + "\n")