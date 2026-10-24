import os
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_data, project_folder

(X_train, y_train, X_val, y_val, X_test, y_test), seed = load_data()

with open(os.path.join(project_folder, "configs", "frozen_tree.json")) as f:
    cfg = json.load(f)
depth, leaf = cfg["max_depth"], cfg["min_samples_leaf"]

n = len(X_train)                       # 240
B_list = [1, 5, 10, 25, 50, 100, 200]
max_B = max(B_list)

rng = np.random.default_rng([seed, 1])   # separate generator, does not touch the split

lines = []


def say(text):
    print(text)
    lines.append(text)


pred_train = np.zeros((max_B, len(X_train)), dtype=int)
pred_val = np.zeros((max_B, len(X_val)), dtype=int)
pred_test = np.zeros((max_B, len(X_test)), dtype=int)
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


def vote_predict(preds, B):
    """Majority vote of the first B trees. Ties go to the lowest class number."""
    votes = np.zeros((preds.shape[1], 40))
    for b in range(B):
        votes[np.arange(preds.shape[1]), preds[b]] += 1
    return votes.argmax(axis=1)


theory = n * (1 - (1 - 1 / n) ** n)
say(f"frozen tree: max_depth={depth}, min_samples_leaf={leaf}")
say(f"theoretical expected distinct per bootstrap: {theory:.2f} (limit 0.632*n = {0.632 * n:.2f})")
say("")
for B in B_list:
    tr = (vote_predict(pred_train, B) == y_train).mean()
    va = (vote_predict(pred_val, B) == y_val).mean()
    te = (vote_predict(pred_test, B) == y_test).mean()
    say(f"B={B:3d}  train={tr:.4f}  val={va:.4f}  test={te:.4f}  "
        f"avg distinct={distinct[:B].mean():.2f}")

os.makedirs(os.path.join(project_folder, "results"), exist_ok=True)
os.makedirs(os.path.join(project_folder, "logs"), exist_ok=True)
np.savez(os.path.join(project_folder, "results", "q1_bagging.npz"),
         pred_train=pred_train, pred_val=pred_val, pred_test=pred_test,
         in_bag=in_bag, distinct=distinct)
with open(os.path.join(project_folder, "logs", "q1b_bagging.log"), "w") as f:
    f.write("\n".join(lines) + "\n")