import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    faces = fetch_olivetti_faces(shuffle=False)
    X, y = faces.data, faces.target
    s = np.load(os.path.join(project_folder, "data", "split.npz"))
    tr, va, te = s["train"], s["val"], s["test"]
    return (X[tr], y[tr], X[va], y[va], X[te], y[te]), int(s["seed"])


def mask_faces(X):
    X = np.asarray(X).reshape(-1, 64, 64).copy()
    X[:, 24:40, 24:40] = 0
    return X.reshape(len(X), -1)

def vote_counts(preds, mask=None):
    """counts[i, k] = number of trees voting class k for example i.
    mask[b, i] = True means tree b is allowed to vote on example i (used for OOB)."""
    N = preds.shape[1]
    counts = np.zeros((N, 40))
    for b in range(len(preds)):
        rows = np.arange(N) if mask is None else np.where(mask[b])[0]
        counts[rows, preds[b, rows]] += 1
    return counts


def disagreement(counts, B):
    """Average over examples of the fraction of tree pairs that predict different classes."""
    return 1 - ((counts * (counts - 1)).sum(1) / (B * (B - 1))).mean()


def margin(counts, B):
    """Mean of (winning vote fraction - runner-up vote fraction)."""
    f = np.sort(counts / B, axis=1)
    return (f[:, -1] - f[:, -2]).mean()