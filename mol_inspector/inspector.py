"""
Base methods.
"""

from typing import Union, Optional
import shap
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from mol_inspector.utils import load_model
import warnings

import os

os.environ["KERAS_BACKEND"] = "torch"
from keras import Model as KerasModel

# SHAP explainers
explainers = {
    "auto": shap.Explainer,
    "tree": shap.TreeExplainer,
    "permutation": shap.PermutationExplainer,
    "deep": shap.DeepExplainer,
    "gradient": shap.GradientExplainer,
    "kernel": shap.KernelExplainer
}

explainer_type = {
    shap.Explainer: "Auto",
    shap.TreeExplainer: "Tree Explainer",
    shap.PermutationExplainer: "Permutation Explainer",
    shap.DeepExplainer: "Deep Explainer",
    shap.GradientExplainer: "Gradient Explainer",
    shap.KernelExplainer: "Kernel Explainer"
}


class Inspector:
    def __init__(self,
                 model: Union[str, BaseEstimator, KerasModel, XGBClassifier, XGBRegressor],
                 train_feats: Union[np.ndarray, pd.DataFrame],
                 model_type: str = "auto"):
        """
        Initiate the Inspector class.
        :param model: Union[str, BaseEstimator, KerasModel, XGBClassifier, XGBRegressor]
            Give input for the query ML/AI model.
        :param train_feats: Union[np.ndarray, pd.DataFrame]
            The dataset used for training the model.
        :param model_type: str
            Set the model type. To see what is available, SHAP documentation can be seen here:
            https://shap.readthedocs.io/en/latest/api.html
        """

        # set instance variables
        self.train_feats = train_feats
        self.model_type = model_type

        # load model
        self.model = load_model(model) if isinstance(model, str) else model
        self._validate()

        # pull explainer
        self.explainer = self._choose_explainer()

        # signify which explainer is pulled
        print(f"Inspector Initialized {explainer_type.get(type(self.explainer), str(type(self.explainer)))}")

    def values(self, test_feats: Union[np.ndarray, pd.DataFrame]):
        """
        Calculate SHAP values.
        :param test_feats: Union[np.ndarray, pd.DataFrame]
            The dataset used for the model prediction. It can be either the test split or the prediction dataset.
        :return:
        """
        values = self.explainer(test_feats)
        return values

    def to_df(self, shap_values, shap_type: str = "values"):
        """
        For users who want to create their own plots. The SHAP values can be converted into a pd.DataFrame. Several
        options are available: 'values' - Main SHAP values for plotting/analysis, 'data' - input features, 'base values'
        - the model's expected output, 'feature names' - names for feature input, and 'output names' for class output
        labeling. Only values and data will output as pd.DataFrame.
        data - original data passed into explainer
        base_values - expected model output
        :param shap_values:
            Input calculated SHAP values. Can accept 'values', 'data', 'base values', 'feature names', or
            'output names'.
        :param shap_type: str
            The type of SHAP values to output.
        :return:
        """

        # available SHAP value types
        value_types = {
            "values": shap_values.values,
            "data": shap_values.data,
            "base values": shap_values.base_values,
            "feature names": shap_values.feature_names,
            "output names": shap_values.output_names
        }

        if shap_type == "base values" or shap_type == "feature names" or shap_type == "output names":
            output = value_types.get(shap_type)
        elif shap_type == "values" or shap_type == "data":
            # get column name and values
            names = self.train_feats.columns
            values = value_types.get(shap_type)
            # convert to pd.DataFrame
            output = pd.DataFrame(values, columns=names)
        else:
            raise ValueError(
                "shap_type incorrect. Only 'values', 'data', 'base values', 'feature names', 'output names' available!")

        return output

    def _validate(self):
        """validate model"""
        if self.model is None:
            raise ValueError("A trained model or a path to a saved model is needed!")
        if self.train_feats is None:
            raise ValueError("Training features must be provided!")

    def _choose_explainer(self):
        """pull explainer from the dictionary"""
        # use model predict_probab if available
        has_proba = hasattr(self.model, "predict_proba")
        model_func = self.model.predict_proba if has_proba else self.model.predict

        # pull shap.Explainer depending on the model type.
        if self.model_type == "auto":
            if isinstance(self.model, (XGBClassifier, XGBRegressor, RandomForestClassifier)):
                explainer_class = shap.TreeExplainer
                return explainer_class(self.model)
            elif isinstance(self.model, (KerasModel, KNeighborsClassifier, KNeighborsRegressor, LogisticRegression)):
                explainer_class = shap.PermutationExplainer
                return explainer_class(model_func, self.train_feats)
            else:
                explainer_class = shap.KernelExplainer
                warnings.warn("Using slow KernelExplainer fallback. Consider specifying a better model_type.")
                return explainer_class(model_func, self.train_feats)
        # pull from explainers dict
        elif self.model_type in explainers:
            explainer_class = explainers[self.model_type]
            if explainer_class in [shap.TreeExplainer, shap.Explainer]:
                return explainer_class(self.model)
            else:
                return explainer_class(model_func, self.train_feats)
        else:
            raise ValueError(f"Only 'auto', 'tree', 'permutation', 'kernel', 'deep', and 'gradient' are supported!")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
