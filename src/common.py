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