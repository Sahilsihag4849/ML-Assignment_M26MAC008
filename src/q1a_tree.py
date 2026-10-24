import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

# paths (works wherever you run it from)
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load faces and the SAVED split (never rebuild the split)
faces = fetch_olivetti_faces(shuffle=False)
X = faces.data
y = faces.target

split = np.load(os.path.join(project_folder, "data", "split.npz"))
train_idx = split["train"]
val_idx = split["val"]
test_idx = split["test"]
seed = int(split["seed"])

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

depths = [2, 4, 8, 16, None]
leaf_sizes = [1, 2, 4]

lines = []   # everything we print also goes into the log file


def say(text):
    print(text)
    lines.append(text)


# ---- the grid: 5 depths x 3 leaf sizes = 15 trees ----
results = []
for d in depths:
    for m in leaf_sizes:
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=seed)
        tree.fit(X_train, y_train)
        val_acc = tree.score(X_val, y_val)
        n_leaves = tree.get_n_leaves()
        results.append((d, m, val_acc, n_leaves))
        say(f"max_depth={d}, min_samples_leaf={m}, val_acc={val_acc:.4f}, leaves={n_leaves}")

# ---- select using VALIDATION accuracy only ----
# highest val accuracy wins; if tied, the tree with fewer leaves wins
best = sorted(results, key=lambda r: (-r[2], r[3]))[0]
best_depth, best_leaf = best[0], best[1]
say("")
say(f"SELECTED (frozen): max_depth={best_depth}, min_samples_leaf={best_leaf}")

# ---- refit the frozen tree and report everything ----
tree = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=best_leaf, random_state=seed)
tree.fit(X_train, y_train)

say(f"train accuracy: {tree.score(X_train, y_train):.4f}")
say(f"val accuracy:   {tree.score(X_val, y_val):.4f}")
say(f"test accuracy:  {tree.score(X_test, y_test):.4f}")

# leaf statistics: a node is a leaf if it has no left child (-1)
t = tree.tree_
is_leaf = t.children_left == -1
samples_in_leaves = t.n_node_samples[is_leaf]    # training samples in each leaf

say(f"number of leaves: {is_leaf.sum()}")
say(f"mean training samples per leaf: {samples_in_leaves.mean():.3f}")
say(f"fraction of leaves with exactly 1 training sample: {(samples_in_leaves == 1).mean():.3f}")

# save the log
os.makedirs(os.path.join(project_folder, "logs"), exist_ok=True)
with open(os.path.join(project_folder, "logs", "q1a_tree.log"), "w") as f:
    f.write("\n".join(lines) + "\n")