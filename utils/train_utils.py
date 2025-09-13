import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .db_utils import fetch_all

MODEL_PATH = "models/audio_model.pkl"


def train_model(progress_callback=None):
    data = fetch_all()
    if not data:
        return None

    X, y = [], []
    for idx, (q, a, mfcc_blob) in enumerate(data):
        mfcc = np.frombuffer(mfcc_blob, dtype=np.float32)
        X.append(mfcc)
        y.append(a)

        if progress_callback:
            progress_callback((idx + 1) / len(data))

    # Train a simple classifier
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except:
        return None
