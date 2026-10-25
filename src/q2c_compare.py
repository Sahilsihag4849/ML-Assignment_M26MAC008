import os
import json
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_data, project_folder, vote_counts
from samme import fit_samme, samme_scores

(Xtr, ytr, Xva, yva, Xte, yte), seed = load_data()
K, n = 40, len(ytr)
tree_cfg = json.load(open(os.path.join(project_folder, "configs", "frozen_tree.json")))
boost_cfg = json.load(open(os.path.join(project_folder, "configs", "final_boosting.json")))
depth, rounds = boost_cfg["depth"], boost_cfg["rounds"]
sets = [(Xtr, ytr), (Xva, yva), (Xte, yte)]      # train, validation, clean test
log = []

#part 1: bagging vs boosting
for L in [25, 50, 100]:
    start = time.time()
    rng = np.random.default_rng([seed, 1])        # same random numbers as Q1b
    trees = []
    for b in range(L):
        s = rng.integers(0, n, size=n)            # bootstrap sample
        trees.append(DecisionTreeClassifier(**tree_cfg, random_state=seed + b).fit(Xtr[s], ytr[s]))
    t_bag = time.time() - start
    acc = [(vote_counts(np.array([t.predict(X) for t in trees])).argmax(1) == y).mean() for X, y in sets]
    log.append(f"bagging  L={L:3d} train={acc[0]:.4f} val={acc[1]:.4f} test={acc[2]:.4f} time={t_bag:.1f}s")

    start = time.time()
    learners, alphas, _ = fit_samme(Xtr, ytr, K, L, depth, seed)
    t_boost = time.time() - start
    acc = [(samme_scores(learners, alphas, X, K).argmax(1) == y).mean() for X, y in sets]
    log.append(f"boosting L={L:3d} train={acc[0]:.4f} val={acc[1]:.4f} test={acc[2]:.4f} time={t_boost:.1f}s")
    print(log[-2]); print(log[-1])

# part 2: the training face with the largest boosting weight
saved = np.load(os.path.join(project_folder, "results", "q2_boosting.npz"))
W = saved[f"d{depth}_W"][:rounds]                 # W[t, i] = weight of image i in round t
miss = ~saved[f"d{depth}_correct"][:rounds]       # miss[t, i] = image i wrong in round t

face = int(W.max(axis=0).argmax())                # image with the biggest weight in any round
person = int(ytr[face])
log.append("")
log.append(f"heaviest face: training image {face}, person {person}")
log.append(f"weight: start {W[0, face]:.4f}, largest {W[:, face].max():.4f} "
           f"(round {int(W[:, face].argmax()) + 1})")
log.append(f"wrong in {int(miss[:, face].sum())} of {rounds} rounds")
log.append("history (X = wrong, . = right): " + "".join("X" if m else "." for m in miss[:, face]))

ratio = W[1:, face] / W[:-1, face]                # weight change from one round to the next
log.append(f"weight change after a miss: x{ratio[miss[:-1, face]].mean():.2f}")
log.append(f"weight change after a hit:  x{ratio[~miss[:-1, face]].mean():.2f}")

dist = np.linalg.norm(Xtr - Xtr[face], axis=1)    # distance to every training image
dist[face] = np.inf                               # ignore itself
log.append(f"nearest image of the same person:    {dist[ytr == person].min():.2f}")
log.append(f"nearest image of another person:     {dist[ytr != person].min():.2f}")

print("\n".join(log))
open(os.path.join(project_folder, "logs", "q2c.log"), "w").write("\n".join(log) + "\n")