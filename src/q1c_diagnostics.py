import os
import json
import numpy as np
import matplotlib.pyplot as plt
from common import load_data, project_folder, vote_counts, disagreement, margin

(_, ytr, _, yva, _, _), seed = load_data()
r = np.load(os.path.join(project_folder, "results", "q1_bagging.npz"))
Bs = [1, 5, 10, 25, 50, 100, 200]

rows, lines = [], []
for B in Bs:
    c = vote_counts(r["pred_val"][:B])
    oc = vote_counts(r["pred_train"][:B], ~r["in_bag"][:B])   # only trees that missed the image
    has = oc.sum(1) > 0
    row = ((c.argmax(1) == yva).mean(), (oc.argmax(1)[has] == ytr[has]).mean(),
           disagreement(c, B) if B > 1 else np.nan, margin(c, B))
    rows.append(row)
    lines.append(f"B={B:3d} val={row[0]:.4f} oob={row[1]:.4f} (covered {has.sum()}/240) "
                 f"disagreement={row[2]:.4f} margin={row[3]:.4f}")
rows = np.array(rows)

best_B = Bs[int(np.argmax(rows[:, 0]))]      # argmax returns the first max, so ties go to smaller B
lines.append(f"SELECTED B={best_B}")
print("\n".join(lines))

fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, col, name in zip(ax, [1, 2, 3], ["OOB accuracy", "disagreement (val)", "vote margin (val)"]):
    a.plot(Bs, rows[:, col], "o-"); a.set_xscale("log"); a.set_xlabel("B"); a.set_ylabel(name)
plt.tight_layout()
plt.savefig(os.path.join(project_folder, "figures", "q1c_diagnostics.png"), dpi=150)

open(os.path.join(project_folder, "logs", "q1c_diagnostics.log"), "w").write("\n".join(lines) + "\n")
json.dump({"B": best_B}, open(os.path.join(project_folder, "configs", "final_bagging.json"), "w"))