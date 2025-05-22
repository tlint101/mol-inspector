"""
plot methods
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union


class Plots:
    def __init__(self, shap_values, train_feats: Union[np.ndarray, pd.DataFrame],
                 test_feats: Union[np.ndarray, pd.DataFrame]):
        """
        Initialize the Plots object. This requires three arguments.
        :param shap_values:
            Input calculated SHAP values.
        :param train_feats:
        :param test_feats:
        """
        self.shap_values = shap_values
        self.train_feats = train_feats
        self.test_feats = test_feats

    def summary_plot(self, max_display: int = 20, palette: str = "coolwarm", alpha: float = 0.6, size: int = 4,
                     jitter: float = 0.2, figsize: tuple = (8, 8), savefig: str = None, **kwargs):
        """
        Generate a summary plot of the mean SHAP value for the model output.
        :param max_display: int
            Set the number of features to display. Features are ordered by SHAP value from most to least impactful.
        :param palette: str
            Set the color palette to use. Only color palettes available in seaborn are supported.
        :param alpha: float
            Set transparency of the plot.
        :param size: float
            set the size of the plot points.
        :param jitter: float
            set the amount of jitter.
        :param figsize: tuple
            Set the figure size of the plot.
        :param savefig: str
            Indicate path to save the image.
        :param kwargs:
            Additional seaborn plot options
        :return:
        """

        shap_df = self._to_df(self.shap_values)
        train_feats_df = self._to_df(self.shap_values, shap_type="data")

        # shap = self.shap_values.values
        # train_feats = self.shap_values.data

        # melt and combine dfs for plotting
        shap_melt = shap_df.melt(var_name='Feature', value_name='SHAP value')
        train_melt = train_feats_df.melt(var_name='Feature', value_name='Feature value')
        combined = pd.concat([shap_melt, train_melt['Feature value']], axis=1)

        # sort by feature importance (mean |SHAP|)
        feat_order = shap_df.abs().mean().sort_values(ascending=False).index

        # plot variables
        title = "SHAP Summary"
        x_title = "SHAP Value (Impact on Model Output)"
        y_title = "Features"
        legend_title = "Feature Value"

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        plt.grid(False)
        ax = sns.stripplot(data=combined, x="SHAP value", y="Feature", hue='Feature value',
                           order=feat_order[:max_display], palette=palette, dodge=False, alpha=alpha, size=size,
                           jitter=jitter, **kwargs)

        # # color bar
        ax.get_legend().remove()
        # normalize feature value
        vmin = combined['Feature value'].min()
        vmax = combined['Feature value'].max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # apply palette
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])  # Required for matplotlib to render colorbar

        # add colorbar
        cbar = ax.figure.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)

        # add low and high ticks
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['Low', 'High'])

        # cbar labelsize
        cbar.set_label("Feature Value", fontsize=10)
        cbar.ax.tick_params(labelsize=10)

        # plot features
        plt.axvline(0.0, color='gray', linewidth=1, linestyle='--')  # line
        plt.title(title, fontsize=18)
        plt.xlabel(x_title, fontsize=14)
        plt.ylabel(y_title, fontsize=14)
        # plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.close()

    def bar(self, index: int = None, bar_color: str = None, text_color: str = None, drop_sum: bool = False, **kwargs):
        pass

    def _to_df(self, shap_values, shap_type: str = "values"):
        """
        Support function to convert SHAP values into pd.DataFrame for plotting. Method is adapted from the method in
        inspector.py
        """

        # available SHAP value types
        value_types = {
            "values": shap_values.values,
            "data": shap_values.data,
            "base values": shap_values.base_values,
            "feature names": shap_values.feature_names,
            "output names": shap_values.output_names
        }

        # get column name and values
        names = self.train_feats.columns
        values = value_types.get(shap_type)
        # convert to pd.DataFrame
        output = pd.DataFrame(values, columns=names)

        return output


class MolInspector:
    def __init__(self, smi: str = None):
        self.smi = smi


if __name__ == "__main__":
    import doctest

    doctest.testmod()
