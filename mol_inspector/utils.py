import pickle
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras


def load_model(model_path):
    """load model"""
    global loaded
    try:
        loaded = keras.saving.load_model(filepath=model_path)
    except Exception:
        try:
            with open(model_path, "rb") as f:
                loaded = pickle.load(f)
        except Exception as e:
            print(e)

    return loaded


if __name__ == "__main__":
    import doctest

    doctest.testmod()
