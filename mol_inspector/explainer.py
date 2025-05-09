"""
Base methods.
"""

import shap
import pandas as pd
from sklearn.base import BaseEstimator
from mol_inspector.utils import load_model


class Explainer:
    def __init__(self, model_path, features, model_type: str = 'auto'):
        # load model
        self.model = load_model(model_path)

        # dataset and SHAP explainer
        self.features = features
        self.model_type = model_type
        self.explainer = None

    def initialize(self):
        """Try to automatically choose the correct SHAP explainer."""
        try:
            if self.model_type == "auto":
                self.explainer = shap.Explainer(self.model, self.features)
            elif self.model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model, self.features)
            elif self.model_type == "linear":
                self.explainer = shap.LinearExplainer(self.model, self.features)
            elif self.model_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, self.features)
            elif self.model_type == "kernel":
                self.explainer = shap.KernelExplainer(self.model.predict, self.features)
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")
        except Exception as e:
            print(f"[SHAP Error] Explainer creation failed: {e}")
        return self.explainer

    def explain(self, X):
        """Explain a dataset or a single sample."""
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call `auto()` first.")
        try:
            shap_values = self.explainer(X)
            return shap_values
        except Exception as e:
            print(f"[SHAP Error] Explanation failed: {e}")
            return None

    def summary_plot(self, shap_values, X=None, plot_type="bar", **kwargs):
        """Optional SHAP summary plot."""
        if X is None:
            X = self.features
        try:
            shap.summary_plot(shap_values, X, plot_type=plot_type, **kwargs)
        except Exception as e:
            print(f"[SHAP Error] Plotting failed: {e}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
