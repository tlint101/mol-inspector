"""
Base methods.
"""

import shap
import pandas as pd
from sklearn.base import BaseEstimator


class Explainer:
    def __init__(self, model_path, features):
        # load model
        from mol_inspector.utils import load_model
        self.model = load_model(model_path)

        # dataset and SHAP explainer
        self.dataset = features
        self.explainer = None

    def auto(self):
        try:
            self.explainer = shap.Explainer(self.model, self.dataset)
        except Exception as e:
            print(f"Auto Explainer failed: {e}")
        return self.explainer

    def tree(self):
        model = self.model
        features = self.dataset

        shap.TreeExplainer()

    def deep_networks(self):
        model = self.model
        features = self.dataset

        shap.DeepExplainer()

    def inspect(self):
        global type
        model = self.model
        dataset = self.dataset

        type = type(model).__name__.lower()

        print(type)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
