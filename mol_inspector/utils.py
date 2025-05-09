import pickle
import os
from xgboost import XGBClassifier

os.environ["KERAS_BACKEND"] = "torch"

import keras


def load_model(model_path):
    """load model"""
    global loaded
    try:
        # keras
        loaded = keras.saving.load_model(filepath=model_path)
    except Exception:
        try:
            # pickle
            with open(model_path, "rb") as f:
                loaded = pickle.load(f)
        except Exception:
            try:
                # json or model
                loaded = XGBClassifier()
                loaded.load_model(model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
                loaded = None

    return loaded


if __name__ == "__main__":
    import doctest

    doctest.testmod()
