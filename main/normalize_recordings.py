import numpy as np, glob, os

SEQ_LEN = 30
GESTURES = os.listdir("gesture_data")
print(GESTURES)
print(len(GESTURES))
X, y = [], []

for label, g in enumerate(GESTURES):
    for file in glob.glob(f"gesture_data/{g}/*.npy"):
        seq = np.load(file)
        # pad or truncate to SEQ_LEN
        if len(seq) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(seq), seq.shape[1]))
            seq = np.concatenate([seq, pad])
        else:
            seq = seq[:SEQ_LEN]
        X.append(seq)
        y.append(label)

np.save("npy/X.npy", np.array(X))
np.save("npy/y.npy", np.array(y))