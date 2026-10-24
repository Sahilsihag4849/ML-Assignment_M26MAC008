import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# load the faces
faces = fetch_olivetti_faces(shuffle=False)
X = faces.data      # 400 images, each flattened to 4096 pixels
y = faces.target    # person number (0 to 39) for each image

# make the seed from the roll number (digits only)
roll_number = "M26MAC008"
digits = ""
for ch in roll_number:
    if ch.isdigit():
        digits = digits + ch
seed = int(digits)
print("seed:", seed)

rng = np.random.default_rng(seed)

# split each person's 10 images into 6 train, 2 val, 2 test
train_idx = []
val_idx = []
test_idx = []

for person in range(40):
    start = person * 10
    idx = np.arange(start, start + 10)   # the 10 image numbers of this person
    rng.shuffle(idx)                     # shuffle them
    train_idx.extend(idx[0:6])
    val_idx.extend(idx[6:8])
    test_idx.extend(idx[8:10])

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

# quick checks
print("sizes:", len(train_idx), len(val_idx), len(test_idx))
assert len(train_idx) == 240
assert len(val_idx) == 80
assert len(test_idx) == 80
assert (np.bincount(y[train_idx]) == 6).all()   # every person has 6 train images
assert (np.bincount(y[val_idx]) == 2).all()
assert (np.bincount(y[test_idx]) == 2).all()
assert len(set(train_idx) | set(val_idx) | set(test_idx)) == 400   # no overlap

# save the split
import os
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(project_folder, "data", "split.npz")
np.savez(save_path, train=train_idx, val=val_idx, test=test_idx, seed=seed)
print("split saved")

# make the masked test set (just to look at it)
X_test = X[test_idx]
images = X_test.reshape(80, 64, 64).copy()   # back to 64x64 pictures
images[:, 24:40, 24:40] = 0                  # black square in the middle
X_test_masked = images.reshape(80, 4096)     # flatten again

import matplotlib.pyplot as plt
plt.imshow(images[0], cmap="gray")
plt.show()