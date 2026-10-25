import os
import json
import glob
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_data, project_folder, disagreement
from q3_refits import CONFIGS

K = 40
(Xtr, ytr, _, _, Xte, yte), seed = load_data()
B_final = json.load(open(os.path.join(project_folder, "configs", "final_bagging.json")))["B"]

files = sorted(glob.glob(os.path.join(project_folder, "results", "q3", "rep_*.npz")))
reps = [np.load(f) for f in files]
log = [f"{len(reps)} repetitions loaded"]

Y = np.zeros((len(yte), K))
Y[np.arange(len(yte)), yte] = 1                                    # one-hot targets e_i


def stack(key, scale=1):
    return np.stack([rep[key] for rep in reps]) / scale            # shape (R, N, K)


def measure(P):
    pbar = P.mean(axis=0)                                          # mean class vector over repetitions
    bias2 = ((pbar - Y) ** 2).sum(axis=1).mean()
    variance = ((P - pbar) ** 2).sum(axis=2).mean()
    acc = (P.argmax(axis=2) == yte).mean(axis=1)                   # test accuracy of each refit
    assert np.isclose(((P - Y) ** 2).sum(axis=2).mean(), bias2 + variance)   # sanity: error = bias^2 + variance
    return bias2, variance, acc.mean(), acc.std(ddof=1)


# ---------- Q3(a) ----------
log.append("\nQ3(a): selected single tree, bagging, boosting (clean test set)")
for name, key, scale in [("single tree", "tree_selected_clean", 1),
                         ("bagging", "bag_final_clean", B_final),
                         ("boosting", "boost_clean", 1)]:
    b2, var, m, s = measure(stack(key, scale))
    log.append(f"{name:12s} bias2={b2:.4f} variance={var:.4f} acc mean={m:.4f} std={s:.4f}")

# ---------- Q3(b) and (c): three settings, single tree vs 100-tree bagging ----------
log.append("\nQ3(b,c): per setting")
for name, cfg in CONFIGS.items():
    one = measure(stack(f"tree_{name}_clean"))
    bag = measure(stack(f"bag100_{name}_clean", 100))
    tree = DecisionTreeClassifier(**cfg, random_state=seed).fit(Xtr, ytr).tree_    # on the ORIGINAL training set
    sizes = tree.n_node_samples[tree.children_left == -1]
    dis = np.mean([disagreement(c, 100) for c in stack(f"bag100_{name}_clean")])
    log.append(f"{name:8s} {cfg}")
    log.append(f"   single: bias2={one[0]:.4f} var={one[1]:.4f} acc={one[2]:.4f}+-{one[3]:.4f}")
    log.append(f"   bag100: bias2={bag[0]:.4f} var={bag[1]:.4f} acc={bag[2]:.4f}+-{bag[3]:.4f}")
    log.append(f"   variance removed={one[1] - bag[1]:.4f} ({100 * (one[1] - bag[1]) / one[1]:.1f}%), "
               f"accuracy gain={bag[2] - one[2]:+.4f}")
    log.append(f"   leaves={len(sizes)}, mean samples/leaf={sizes.mean():.2f}, "
               f"single-sample leaves={(sizes == 1).mean():.3f}, mean tree disagreement={dis:.4f}")

# ---------- Q3(d): final table ----------
log.append("\nQ3(d): final table (means over the refits)")
log.append("model         clean_acc masked_acc | clean bias2  var | masked bias2  var")
for name, base, scale in [("single tree", "tree_selected", 1), ("bagging", "bag_final", B_final),
                          ("boosting", "boost", 1)]:
    c = measure(stack(f"{base}_clean", scale))
    m = measure(stack(f"{base}_masked", scale))
    log.append(f"{name:12s}  {c[2]:.4f}    {m[2]:.4f}    | {c[0]:.4f}  {c[1]:.4f} | {m[0]:.4f}  {m[1]:.4f}")

print("\n".join(log))
open(os.path.join(project_folder, "logs", "q3_analysis.log"), "w").write("\n".join(log) + "\n")