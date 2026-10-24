import os
import json
from sklearn.tree import DecisionTreeClassifier
from common import load_data, project_folder

(X_train, y_train, X_val, y_val, X_test, y_test), seed = load_data()

lines = []


def say(text):
    print(text)
    lines.append(text)


def leaf_samples(tree):
    t = tree.tree_
    return t.n_node_samples[t.children_left == -1]   # training samples in each leaf


results = []
for d in [2, 4, 8, 16, None]:
    for m in [1, 2, 4]:
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=seed)
        tree.fit(X_train, y_train)
        ls = leaf_samples(tree)
        val_acc = tree.score(X_val, y_val)
        results.append((d, m, val_acc, len(ls)))
        say(f"max_depth={d}, min_samples_leaf={m}, train_acc={tree.score(X_train, y_train):.4f}, "
            f"val_acc={val_acc:.4f}, leaves={len(ls)}, mean/leaf={ls.mean():.2f}, "
            f"single-sample leaves={(ls == 1).mean():.3f}")

# select by VALIDATION accuracy only; ties go to the tree with fewer leaves
best_depth, best_leaf, _, _ = sorted(results, key=lambda r: (-r[2], r[3]))[0]
say("")
say(f"SELECTED (frozen): max_depth={best_depth}, min_samples_leaf={best_leaf}")

tree = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=best_leaf, random_state=seed)
tree.fit(X_train, y_train)
ls = leaf_samples(tree)
say(f"train accuracy: {tree.score(X_train, y_train):.4f}")
say(f"val accuracy:   {tree.score(X_val, y_val):.4f}")
say(f"test accuracy:  {tree.score(X_test, y_test):.4f}")
say(f"number of leaves: {len(ls)}")
say(f"mean training samples per leaf: {ls.mean():.3f}")
say(f"fraction of leaves with exactly 1 training sample: {(ls == 1).mean():.3f}")

os.makedirs(os.path.join(project_folder, "logs"), exist_ok=True)
os.makedirs(os.path.join(project_folder, "configs"), exist_ok=True)
with open(os.path.join(project_folder, "logs", "q1a_tree.log"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(project_folder, "configs", "frozen_tree.json"), "w") as f:
    json.dump({"max_depth": best_depth, "min_samples_leaf": best_leaf}, f)