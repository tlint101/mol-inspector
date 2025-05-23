"""
plot methods
"""

import pandas as pd
import numpy as np
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
            Set the save path to save the image.
        :param kwargs:
            Additional seaborn plot options
        :return:
        """
        # process dataset
        combined, feat_order = self._process_shap_values(shap_type="value")

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
        plt.title("SHAP Summary", fontsize=18)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        # plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.close()

    def bar(self, max_display: int = 10, color: str = "steelblue", label: bool = True,
            figsize: tuple = (8, 8), savefig: str = None, **kwargs):
        """
        Generate a bar plot of the mean SHAP values for the models. Data will be argued from largest to lowest, with
        the top 10 features shown by default.
        :param max_display: int
            Set the number of features to be shown on the plot.
        :param color: str
            Set the color of the bar plot. Color can be given as a name or a hex color code.
        :param label: bool
            Annotate the plot with the mean SHAP value.
        :param figsize: tuple
            Set the figure size of the plot.
        :param savefig: str
            Set the save path for the figure.
        :param kwargs:
            Additional kwargs for seaborn plots.
        :return:
        """
        # process dataset
        combined = self._process_shap_values(shap_type="mean")
        combined = combined.head(max_display)

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        plt.grid(False)
        ax = sns.barplot(data=combined, x="Mean(SHAP Value)", y="Feature", color=color,
                         **kwargs)

        # add label to bar
        if label is True:
            for i, (value, feature) in enumerate(zip(combined["Mean(SHAP Value)"], combined["Feature"])):
                plot_label = f"+{value:.3f}" if value > 0 else f"{value:.3f}"
                ax.text(value + 0.001, i, plot_label, va='center', fontsize=12)

            # increase plot boarder
            max_val = combined["Mean(SHAP Value)"].max()
            plt.xlim(0, max_val * 1.15)

        # Customize fonts
        ax.set_title("SHAP Feature Importance", fontsize=16)
        ax.set_xlabel("Mean |SHAP value|", fontsize=13)
        ax.set_ylabel("Feature", fontsize=13, labelpad=10)
        plt.tight_layout()
        plt.show()
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.close()

    def waterfall(self):
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
        }

        # get column name and values
        names = self.train_feats.columns
        values = value_types.get(shap_type)
        # convert to pd.DataFrame
        output = pd.DataFrame(values, columns=names)

        return output

    def _process_shap_values(self, shap_type='value'):
        """support function to process shap_values into data for plotting."""
        # set SHAP and feature data
        if shap_type == 'value':
            shap_df = self._to_df(self.shap_values)
            train_feats_df = self._to_df(self.shap_values, shap_type="data")
            # melt and combine dfs for plotting
            shap_melt = shap_df.melt(var_name='Feature', value_name='SHAP value')
            train_melt = train_feats_df.melt(var_name='Feature', value_name='Feature value')
            combined = pd.concat([shap_melt, train_melt['Feature value']], axis=1)
            # sort by feature importance (mean |SHAP|)
            feat_order = shap_df.abs().mean().sort_values(ascending=False).index
            return combined, feat_order

        # get mean of SHAP values
        elif shap_type == 'mean':
            shap_values = self.shap_values.values
            feat_names = self.shap_values.feature_names
            # calculate mean
            mean = np.abs(shap_values).mean(axis=0)
            # generate df and sort values
            df = pd.DataFrame({"Feature": feat_names, "Mean(SHAP Value)": mean})
            df = df.sort_values(by="Mean(SHAP Value)", ascending=False)
            return df
        else:
            return None


class MolInspector:
    def __init__(self, smi: str = None):
        self.smi = smi


if __name__ == "__main__":
    import doctest

    doctest.testmod()
