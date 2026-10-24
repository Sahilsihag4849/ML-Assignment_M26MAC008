import numpy as np
from sklearn.tree import DecisionTreeClassifier


def samme_step(w, pred, y, K):
    """One SAMME update. Returns (eps, alpha, new_weights, status)."""
    eps = w[pred != y].sum() / w.sum()           # weighted error
    if eps >= 1 - 1 / K:                         # not better than random K-class guessing
        return eps, 0.0, w, "stop_random"
    status = "ok"
    if eps <= 1e-10:                             # perfect learner: alpha would be infinite
        eps, status = 1e-10, "stop_perfect"
    alpha = np.log((1 - eps) / eps) + np.log(K - 1)
    new_w = w * np.exp(alpha * (pred != y))      # only misclassified samples are boosted
    return eps, alpha, new_w / new_w.sum(), status


def fit_samme(X, y, K, T, depth, seed):
    """Run up to T SAMME rounds. Returns learners, alphas, and a per-round history."""
    w = np.full(len(y), 1 / len(y))
    learners, alphas, hist = [], [], []
    for t in range(T):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed + t)
        tree.fit(X, y, sample_weight=w)
        pred = tree.predict(X)
        eps, alpha, w_new, status = samme_step(w, pred, y, K)
        hist.append(dict(eps=eps, alpha=alpha, status=status, w=w.copy(), correct=(pred == y)))
        if status == "stop_random":              # discard this learner and stop
            break
        learners.append(tree)
        alphas.append(alpha)
        w = w_new
        if status == "stop_perfect":             # keep it, but nothing is left to boost
            break
    return learners, np.array(alphas), hist


def samme_scores(learners, alphas, X, K):
    """S[i, k] = sum of alpha_t over learners voting class k. Divide by alphas.sum() for q_k."""
    S = np.zeros((len(X), K))
    for tree, a in zip(learners, alphas):
        S[np.arange(len(X)), tree.predict(X)] += a
    return S