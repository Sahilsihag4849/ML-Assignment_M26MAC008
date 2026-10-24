import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import project_folder
from samme import samme_step, fit_samme, samme_scores

K = 3
X = np.array([[1.0], [1.2], [1.4], [2.0], [2.2], [2.4], [3.0], [3.2], [3.4]])   # 9 samples, 1 feature
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
n = len(y)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

lines = []


def say(text):
    print(text)
    lines.append(text)


# ---- one complete SAMME round, logged step by step ----
w = np.full(n, 1 / n)
tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
pred = tree.predict(X)
eps, alpha, w_new, status = samme_step(w, pred, y, K)

say(f"true labels:        {y}")
say(f"initial weights:    {w}")
say(f"weak learner pred:  {pred}")
say(f"misclassified:      {(pred != y).astype(int)}")
say(f"weighted error:     {eps:.4f}   (random 3-class guessing would give {1 - 1 / K:.4f})")
say(f"alpha:              log((1-eps)/eps) + log(K-1) = {alpha:.4f}")
say(f"updated weights:    {w_new}   (sum = {w_new.sum():.4f}, status = {status})")

# ---- the same function drives a short full run ----
learners, alphas, hist = fit_samme(X, y, K, T=3, depth=1, seed=0)
S = samme_scores(learners, alphas, X, K)
say("")
say(f"3-round run: eps = {[round(float(h['eps']), 4) for h in hist]}, alphas = {alphas.round(4)}")
say(f"ensemble prediction: {S.argmax(1)}  (accuracy {(S.argmax(1) == y).mean():.3f})")
say(f"normalised vote q_k for sample 0: {S[0] / alphas.sum()}")

open(os.path.join(project_folder, "logs", "q2a_toy.log"), "w").write("\n".join(lines) + "\n")
