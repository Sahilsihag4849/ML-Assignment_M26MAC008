import os
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_data, mask_faces, project_folder, vote_counts

(X_train, y_train, X_val, y_val, X_test, y_test), seed = load_data()
cfg = json.load(open(os.path.join(project_folder, "configs", "frozen_tree.json")))
depth, leaf = cfg["max_depth"], cfg["min_samples_leaf"]

n = len(X_train)                         # 240
B_list = [1, 5, 10, 25, 50, 100, 200]
max_B = max(B_list)
rng = np.random.default_rng([seed, 1])   # separate generator, does not touch the split
Xte_m = mask_faces(X_test)               # masked test set (models are trained on clean images only)

pred_train = np.zeros((max_B, len(X_train)), dtype=int)
pred_val = np.zeros((max_B, len(X_val)), dtype=int)
pred_test = np.zeros((max_B, len(X_test)), dtype=int)
pred_test_m = np.zeros((max_B, len(X_test)), dtype=int)
in_bag = np.zeros((max_B, n), dtype=bool)   # was training image j in tree b's sample?
distinct = np.zeros(max_B)

for b in range(max_B):
    boot = rng.integers(0, n, size=n)        # n draws WITH replacement
    in_bag[b, np.unique(boot)] = True
    distinct[b] = len(np.unique(boot))
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed + b)
    tree.fit(X_train[boot], y_train[boot])
    pred_train[b] = tree.predict(X_train)
    pred_val[b] = tree.predict(X_val)
    pred_test[b] = tree.predict(X_test)
    pred_test_m[b] = tree.predict(Xte_m)
    if b % 20 == 0:
        print("tree", b, "of", max_B)

theory = n * (1 - (1 - 1 / n) ** n)
lines = [f"frozen tree: max_depth={depth}, min_samples_leaf={leaf}",
         f"theoretical expected distinct per bootstrap: {theory:.2f} (limit 0.632*n = {0.632 * n:.2f})", ""]
for B in B_list:
    acc = [(vote_counts(p[:B]).argmax(1) == t).mean()
           for p, t in [(pred_train, y_train), (pred_val, y_val), (pred_test, y_test)]]
    lines.append(f"B={B:3d}  train={acc[0]:.4f}  val={acc[1]:.4f}  test={acc[2]:.4f}  "
                 f"avg distinct={distinct[:B].mean():.2f}")
print("\n".join(lines))

np.savez(os.path.join(project_folder, "results", "q1_bagging.npz"),
         pred_train=pred_train, pred_val=pred_val, pred_test=pred_test,
         pred_test_masked=pred_test_m, in_bag=in_bag, distinct=distinct)
open(os.path.join(project_folder, "logs", "q1b_bagging.log"), "w").write("\n".join(lines) + "\n")