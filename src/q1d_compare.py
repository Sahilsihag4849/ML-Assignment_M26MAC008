import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import load_data, mask_faces, project_folder, vote_counts

(Xtr, ytr, _, _, Xte, yte), seed = load_data()
cfg = json.load(open(os.path.join(project_folder, "configs", "frozen_tree.json")))
B = json.load(open(os.path.join(project_folder, "configs", "final_bagging.json")))["B"]
r = np.load(os.path.join(project_folder, "results", "q1_bagging.npz"))

tree = DecisionTreeClassifier(max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
                              random_state=seed).fit(Xtr, ytr)      # same refit as Q1(a)

data = {"clean": (Xte, r["pred_test"][:B]), "masked": (mask_faces(Xte), r["pred_test_masked"][:B])}
lines = [f"frozen tree {cfg}, final B={B}"]

for name, (X, preds) in data.items():
    tp, tconf = tree.predict(X), tree.predict_proba(X).max(1)   # tree confidence = leaf purity
    counts = vote_counts(preds)
    bp, bconf = counts.argmax(1), counts.max(1) / B             # bagging confidence = winning vote fraction
    fixed = np.where((tp != yte) & (bp == yte))[0]
    both = np.where((tp != yte) & (bp != yte))[0]
    broken = ((tp == yte) & (bp != yte)).sum()
    lines.append(f"\n[{name}] tree acc={(tp == yte).mean():.4f} conf={tconf.mean():.4f} | "
                 f"bagging acc={(bp == yte).mean():.4f} conf={bconf.mean():.4f}")
    lines.append(f"[{name}] bagging fixes tree: {len(fixed)}, both wrong: {len(both)}, bagging breaks tree: {broken}")

    picks = [("bag fixes tree", i) for i in fixed[:2]] + [("both wrong", i) for i in both[:2]]
    fig, ax = plt.subplots(1, 4, figsize=(11, 3.4))
    for a in ax:
        a.axis("off")
    for a, (kind, i) in zip(ax, picks):
        top = np.argsort(-counts[i])[:3]
        votes = {int(k): int(counts[i, k]) for k in top}
        lines.append(f"[{name}] {kind}: test face {i}, true={yte[i]}, tree={tp[i]} (conf {tconf[i]:.2f}), "
                     f"bag={bp[i]}, top votes={votes}, votes for true class={int(counts[i, yte[i]])}/{B}")
        a.imshow(X[i].reshape(64, 64), cmap="gray")
        a.set_title(f"{kind}\ntrue {yte[i]} | tree {tp[i]} ({tconf[i]:.2f})\nbag {bp[i]} votes {votes}", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(project_folder, "figures", f"q1d_{name}_cases.png"), dpi=150)
    plt.close()

print("\n".join(lines))
open(os.path.join(project_folder, "logs", "q1d_compare.log"), "w").write("\n".join(lines) + "\n")