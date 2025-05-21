"""
plot methods
"""

import shap
import seaborn as sns
import matplotlib.pyplot as plt


class Plots:
    def __init__(self, shap_values, train_feats, test_feats):
        """

        :param shap_values:
        :param train_feats:
        :param test_feats:
        """
        self.explanation = None
        self.shap_values = shap_values
        self.train_feats = train_feats
        self.features = test_feats

    def summary_plot(self, plot_type: str = None, **kwargs):
        """
        Generate a summary plot of the mean SHAP value for the model output.
        :param plot_type: str
            Set the plot type.
        :param kwargs:
            The kwargs for shap.summary_plot()
        :return:
        """
        values = self.shap_values
        test_feats = self.features
        shap.summary_plot(values, test_feats, plot_type=plot_type, **kwargs)

    def bar(self, index: int = None, bar_color: str = None, text_color: str = None, drop_sum: bool = False, **kwargs):
        test_feats = self.features

        # convert to explination object
        values = self.shap_values
        explanation_value = self._convert_to_explanation(values)

        # for individual data row
        if index:
            explanation_value = explanation_value[0]

        self.explanation = explanation_value

        # plot using shap
        fig = shap.plots.bar(self.explanation, show=False, **kwargs)

        # override shap plot to give control for color palette
        ax = plt.gca()
        for bar in ax.patches:
            bar.set_color(bar_color)

        for text in ax.texts:
            text.set_color(text_color)
        plt.show()

    def force(self, index=0):
        shap.plots.force(self.shap_values[index])

    def _convert_to_explination(self, values):
        explanation = shap.Explanation(
            values=values,
            base_values=None,  # optional, typically from explainer expected_value
            data=self.train_feats.values,
            feature_names=self.train_feats.columns.tolist()
        )

        return explanation


class MolInspector:
    def __init__(self, smi: str = None):
        self.smi = smi


if __name__ == "__main__":
    import doctest

    doctest.testmod()
