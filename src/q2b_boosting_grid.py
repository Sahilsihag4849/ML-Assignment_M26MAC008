import os
import json
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from common import load_data, project_folder
from samme import fit_samme, samme_scores

(Xtr, ytr, Xva, yva, Xte, yte), seed = load_data()
K = 40
depths = [1, 2, 3, 4]
rounds_list = [10, 25, 50, 100]
T = max(rounds_list)

lines, results, store, models = [], [], {}, {}
for d in depths:
    t0 = time.perf_counter()
    learners, alphas, hist = fit_samme(Xtr, ytr, K, T, d, seed)     # one 100-round run per depth
    secs = time.perf_counter() - t0
    models[d] = (learners, alphas)
    store[f"d{d}_eps"] = np.array([h["eps"] for h in hist])
    store[f"d{d}_alpha"] = np.array([h["alpha"] for h in hist])
    store[f"d{d}_W"] = np.array([h["w"] for h in hist])             # weights used to train each round
    store[f"d{d}_correct"] = np.array([h["correct"] for h in hist])
    stops = [h["status"] for h in hist if h["status"] != "ok"]
    lines.append(f"depth={d}: {len(learners)} learners kept, {len(hist)} rounds run, "
                 f"stop status={stops if stops else 'none'}, time={secs:.1f}s")
    for R in rounds_list:                                           # first R learners of the same run
        S = samme_scores(learners[:R], alphas[:R], Xva, K)
        results.append(dict(depth=d, R=R, val=(S.argmax(1) == yva).mean()))
        lines.append(f"  depth={d} rounds={R:3d} val_acc={results[-1]['val']:.4f}")
    print("\n".join(lines[-5:]))

# select by CLEAN VALIDATION accuracy only; ties go to fewer rounds, then smaller depth
best = min(results, key=lambda r: (-r["val"], r["R"], r["depth"]))
d, R = best["depth"], best["R"]
learners, alphas = models[d]
lines.append(f"\nSELECTED (frozen): depth={d}, rounds={R}, val_acc={best['val']:.4f}")
for name, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
    S = samme_scores(learners[:R], alphas[:R], X, K)
    lines.append(f"selected model {name} accuracy: {(S.argmax(1) == y).mean():.4f}")

# ---- plots for the selected model, over its R rounds ----
W = store[f"d{d}_W"][:R]
curves = [("weighted training error", store[f"d{d}_eps"][:R]),
          ("learner weight alpha", store[f"d{d}_alpha"][:R]),
          ("max sample weight", W.max(1)),
          ("weight on 10 heaviest images", np.sort(W, axis=1)[:, -10:].sum(1))]
fig, ax = plt.subplots(1, 4, figsize=(14, 3))
for a, (name, y_vals) in zip(ax, curves):
    a.plot(np.arange(1, len(y_vals) + 1), y_vals, "o-", ms=3)
    a.set_xlabel("round")
    a.set_title(name, fontsize=9)
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(project_folder, "figures", "q2b_selected_history.png"), dpi=150)

print("\n".join(lines[-6:]))
np.savez(os.path.join(project_folder, "results", "q2_boosting.npz"), **store)
os.makedirs(os.path.join(project_folder, "models"), exist_ok=True)
joblib.dump(models, os.path.join(project_folder, "models", "boosting_all_depths.joblib"))
json.dump({"depth": int(d), "rounds": int(R)},
          open(os.path.join(project_folder, "configs", "final_boosting.json"), "w"))
open(os.path.join(project_folder, "logs", "q2b_boosting_grid.log"), "w").write("\n".join(lines) + "\n")