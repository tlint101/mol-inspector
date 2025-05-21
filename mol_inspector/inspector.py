"""
Base methods.
"""

import shap
import pandas as pd
import numpy as np
from typing import Union, Optional
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor, XGBClassifier
from mol_inspector.utils import load_model
import os

os.environ["KERAS_BACKEND"] = "torch"
from keras import Model as KerasModel

explainers = {
    "auto": shap.Explainer,
    "tree": shap.TreeExplainer,
    "permutation": shap.PermutationExplainer,
    "deep": shap.DeepExplainer,
    "gradient": shap.GradientExplainer,
    "kernel": shap.KernelExplainer,
}

explainer_type = {
    shap.Explainer: "Auto",
    shap.TreeExplainer: "Tree Explainer",
    shap.PermutationExplainer: "Permutation Explainer",
    shap.DeepExplainer: "Deep Explainer",
    shap.GradientExplainer: "Gradient Explainer"
}


class Inspector:
    def __init__(self, model: Union[str, BaseEstimator, KerasModel, XGBClassifier, XGBRegressor] = None,
                 train_feats: Union[np.array, pd.DataFrame] = None, model_type: str = "auto"):
        # load model
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model

        # get explainer
        try:
            self.explainer = explainers[model_type]
        except Exception as e:
            raise ValueError(
                f"Model Type {e} not found! Only 'auto', 'tree', 'permutation', 'deep', and 'gradient' supported!")

        # dataset and SHAP explainer
        self.train_feats = train_feats

    def values(self, test_feats: Union[np.array, pd.DataFrame] = None):
        explainer = self.explainer(self.model, test_feats)
        values = explainer(test_feats)
        return values

    def explain(self, model_type: Optional[str] = None, train_features: [np.array, pd.DataFrame] = None):
        # todo Keras with pytorch needs permutation
        # todo for type, set to auto, otherwise use the different explainers
        # todo set the explainers to dictionary
        """Try to automatically choose the correct SHAP explainer."""
        global explainer
        try:
            if model_type is None:
                explainer = explainers[self.model_type]
            elif model_type:
                self.model_type = model_type
                explainer = explainers[model_type]
        except Exception as e:
            raise ValueError(
                f"Model Type {e} not found! Only 'auto', 'tree', 'permutation', 'deep', and 'gradient' supported!")

        self.explainer = explainer

        if self.model_type == "auto":
            explainer_name = self.explainer
        else:
            explainer_name = explainer_type.get(self.explainer)

        return print(f"Initialized {explainer_name}!")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
