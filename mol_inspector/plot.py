"""
plot methods
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors, rdFingerprintGenerator


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
        self.dim_greater_than_2 = False

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

        # set colorbar
        self._set_colorbar(ax, combined, palette)

        # plot features
        plt.axvline(0.0, color='gray', linewidth=1, linestyle='--')  # line
        plt.title("SHAP Summary", fontsize=16)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=13, labelpad=10)
        plt.ylabel("Features", fontsize=13, labelpad=10)
        # plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.close()

    def bar(self, test_idx: int = None, max_display: int = 10, color: Union[str, list[str]] = "lightcoral",
            label: bool = True, figsize: tuple = (8, 8), savefig: str = None, **kwargs):
        """
        Generate a bar plot of the mean SHAP values for the models. Data will be argued from largest to lowest, with
        the top 10 features shown by default.
        :param test_idx: int
            The index to view SHAP values for a specific query from the test set.
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

        # initiate plot
        fig, ax = plt.subplots(figsize=figsize)
        plt.grid(False)

        # process dataset and plot
        if test_idx is not None:
            combined, feat_order = self._process_shap_values(shap_type="value", index=test_idx)
            # get top features
            top_features = feat_order[:max_display]
            combined = combined[combined["Feature"].isin(top_features)].copy()
            # sort by top features
            combined["Feature"] = pd.Categorical(combined["Feature"], categories=top_features, ordered=True)
            plot_data = combined.sort_values("Feature")
            # overwrite str color param
            color = ["lightcoral", "cadetblue"]
            # assign colors to negative/positive values
            palette = [color[0] if v < 0 else color[1] for v in plot_data[plot_data.columns[1]]]
            # plot
            ax = sns.barplot(data=plot_data, x=plot_data.columns[1], y=plot_data.columns[0], hue=plot_data.columns[0],
                             palette=palette, **kwargs)
            plt.axvline(0.0, color='gray', linewidth=1, linestyle='--')  # line
        else:
            combined = self._process_shap_values(shap_type="mean")
            plot_data = combined.head(max_display)
            # plot
            ax = sns.barplot(data=plot_data, x=plot_data.columns[1], y=plot_data.columns[0], color=color,
                             **kwargs)

        # add label to bar
        label_list = []
        if label is True:
            for i, (value, feature) in enumerate(zip(plot_data[plot_data.columns[1]], plot_data[plot_data.columns[0]])):
                plot_label = f"+{value:.3f}" if value > 0 else f"{value:.3f}"

                # shift labels based on sign value
                offset = 0.0001 * (1 if value >= 0 else -1)  # small shift
                ha_alignment = 'left' if value >= 0 else 'right'
                # set text
                ax.text(value + offset, i, plot_label,
                        va='center', ha=ha_alignment, fontsize=12)
                label_list.append(plot_label)

            # increase plot boarder
            # check for negative number
            if any("-" in x for x in label_list):
                min_val = plot_data[plot_data.columns[1]].min()
                max_val = plot_data[plot_data.columns[1]].max()
                range_padding = 0.3 * max(abs(min_val), abs(max_val))
                plt.xlim(min_val - range_padding if min_val < 0 else 0,
                         max_val + range_padding if max_val > 0 else 0)
            else:
                # if all positive
                max_val = plot_data[plot_data.columns[1]].max()
                plt.xlim(0, max_val * 1.15)

        # Customize fonts
        ax.set_title("SHAP Feature Importance", fontsize=16)
        ax.set_xlabel(plot_data.columns[1], fontsize=13, labelpad=10)
        ax.set_ylabel(plot_data.columns[0], fontsize=13, labelpad=10)
        plt.tight_layout()
        plt.show()
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.close()

    def scatterplot(self, feature: str, palette: str = "coolwarm", figsize: tuple = (8, 8), savefig: str = None,
                    **kwargs):

        shap_df = self._to_df(self.shap_values)  # SHAP values (DataFrame)
        data_df = self._to_df(self.shap_values, "data")  # Feature values (DataFrame)

        df = pd.DataFrame({
            "Feature value": data_df[feature],
            "SHAP value": shap_df[feature]
        })

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        plt.grid(False)
        ax = sns.scatterplot(data=df, x="Feature value", y="SHAP value", hue="Feature value", palette=palette,
                             edgecolor=None, **kwargs)

        self._set_colorbar(ax, df, palette)

        plt.title(f"SHAP Values vs Input Value", fontsize=16)
        plt.xlabel(f"Input Value for {feature}", fontsize=13, labelpad=10)
        plt.ylabel(f"SHAP Value for {feature}", fontsize=13, labelpad=10)
        plt.show()
        if savefig:
            plt.savefig(savefig, dpi=300)
        plt.close()

    def _to_df(self, shap_values, shap_type: str = "values", index=None):
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

        # if SHAP values is not 2D due to classification labels
        if values.ndim == 3:
            values = values[:, :, 1]
            self.dim_greater_than_2 = True

        # slice values
        if index is not None:
            values = [values[index]]

        # convert to pd.DataFrame
        output = pd.DataFrame(values, columns=names)

        return output

    def _process_shap_values(self, shap_type='value', index=None):
        """support function to process shap_values into data for plotting."""
        # set SHAP and feature data
        if shap_type == 'value':
            shap_df = self._to_df(self.shap_values, index=index)
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
            if self.dim_greater_than_2:
                mean = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean = np.abs(shap_values).mean(axis=0)

            # generate df and sort values
            df = pd.DataFrame({"Feature": feat_names, "Mean(SHAP Value)": mean})
            df = df.sort_values(by="Mean(SHAP Value)", ascending=False)
            return df
        else:
            return None

    @staticmethod
    def _set_colorbar(ax, combined, palette):
        """support function to set the colorbar"""
        # color bar
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
        cbar.set_label("Feature Value", fontsize=13)
        cbar.ax.tick_params(labelsize=13)


class MolInspector:
    def __init__(self, smi: str = None, fp_type: str = "morgan", n_bits: int = 2048, radius: int = 2,
                 max_path: int = 7, chirality: bool = False, **kwargs):
        self.smi = smi
        self.fp_type = fp_type
        self.bit_info = self._generate_fp(smi=smi, fp_type=fp_type, n_bits=n_bits, radius=radius, max_path=max_path,
                                          chirality=chirality, **kwargs)

    def visualize_fp(self, bit_query, n_mols=3):
        # Convert specific_bit to a list if it's not already a list
        if bit_query is None:
            raise ValueError("Need a bit query for molecular fingerprint!")
        elif not isinstance(bit_query, list):
            bit_query = [bit_query]
        elif isinstance(bit_query, list):
            pass
        else:
            raise ValueError("Maybe an issue with the bit_query?")

        mol = Chem.MolFromSmiles(self.smi)

        # Draw fingerprint bit. The tuple contains molecule, bit query, and bit information.
        try:
            if self.fp_type == "morgan":
                tuple_info = [(mol, bit, self.bit_info) for bit in bit_query]
                fig = Draw.DrawMorganBits(
                    tuple_info,
                    molsPerRow=n_mols,
                    legends=[str(f"Bit: {bit}") for bit in bit_query],
                )
            elif self.fp_type == "rdkit":
                tuple_info = [(mol, bit, self.bit_info) for bit in bit_query]
                fig = Draw.DrawRDKitBits(
                    tuple_info,
                    molsPerRow=n_mols,
                    legends=[str(f"Bit: {bit}") for bit in bit_query],
                )
            else:
                return "Only 'morgan' or 'rdkit' fingerprints are supported!"
        except:
            raise ValueError("Issue with drawing molecule! Check bit_query and nBits!")

        return fig

    def render_bit(self):
        """An interactive method to render query fingerprint bits. Only works in Jupyter Notebooks"""
        from ipywidgets import widgets
        dropdown = widgets.Dropdown(
            options=self.bit_info.keys(),
            description="Select Bit:",
            style={"description_width": "initial"}
        )

        # wrapper method to draw bit
        def draw_bit(index):
            bit_query = index
            return self.visualize_fp(bit_query=bit_query)

        widgets.interact(draw_bit, index=dropdown)

    @staticmethod
    def _generate_fp(smi, fp_type, n_bits, radius, max_path, chirality, **kwargs):
        """Generate fp"""
        # Convert smiles to rdkit mol
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            raise ValueError("SMILES input is not a valid SMILES string!")

        # Generate molecular fingerprint
        if fp_type == "morgan":
            ao = AllChem.AdditionalOutput()
            ao.CollectBitInfoMap()
            fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits,
                                                                     includeChirality=chirality)
            fp_generator.GetFingerprint(mol, additionalOutput=ao)

            # get bit info
            bit_info = ao.GetBitInfoMap()
            return bit_info
        elif fp_type == "rdkit":
            ao = AllChem.AdditionalOutput()
            ao.CollectBitPaths()
            fp_generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits, maxPath=max_path, **kwargs)
            fp_generator.GetFingerprint(mol, additionalOutput=ao)
            bit_info = ao.GetBitPaths()
            return bit_info

        else:
            return f"Issue drawing {fp_type} bits!"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
