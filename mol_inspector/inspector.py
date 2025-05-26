"""
Base methods.
"""

from typing import Union
import shap
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from mol_inspector.utils import load_model

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
    "kernel": shap.KernelExplainer,
    "linear": shap.LinearExplainer,
}


def _get_explainer_type(explainer):
    if isinstance(explainer, shap.TreeExplainer):
        return "Tree Explainer"
    elif isinstance(explainer, shap.PermutationExplainer):
        return "Permutation Explainer"
    elif isinstance(explainer, shap.DeepExplainer):
        return "Deep Explainer"
    elif isinstance(explainer, shap.GradientExplainer):
        return "Gradient Explainer"
    elif isinstance(explainer, shap.KernelExplainer):
        return "Kernel Explainer"
    elif isinstance(explainer, shap.LinearExplainer):
        return "Linear Explainer"
    else:
        return "Unknown (possibly auto-generated)"


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
        print(f"Inspector Initialized", _get_explainer_type(self.explainer))
        if isinstance(self.explainer, shap.PermutationExplainer):
            print("Permutation Explainer may take a while...")

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
            # if SHAP values is not 2D due to classification labels
            if values.ndim == 3:
                values = values[:, :, 1]
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
            # if models are tree/logistic regression
            if isinstance(self.model,
                          (XGBClassifier, XGBRegressor, RandomForestClassifier, LogisticRegression, LinearRegression)):
                return shap.Explainer(self.model, self.train_feats)
            else:
                return shap.Explainer(model_func, self.train_feats, max_evals="auto")

        # pull from explainers dict
        elif self.model_type in explainers:
            explainer_class = explainers[self.model_type]
            if explainer_class in [shap.TreeExplainer, shap.Explainer]:
                return explainer_class(self.model)
            else:
                return explainer_class(model_func, self.train_feats)
        else:
            raise ValueError(
                f"Only 'auto', 'tree', 'permutation', 'kernel', 'deep', 'linear', and 'gradient' are supported!")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
