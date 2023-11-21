''' This module contains the functions to plot the attention matrix for each head and the sensitivity analysis of the model. '''
import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import ticker
from arcana.plots import plots
from arcana.logger import logger

log = logger.get_logger("arcana.plots.analysis_plot_utils")

class ScalarFormatterForceFormat(ticker.ScalarFormatter):
    """ScalarFormatterForceFormat class to format the colorbar labels to scientific notation.
    Inherits from ScalarFormatter class from matplotlib.ticker.

    Args:
        ticker (matplotlib.ticker.ScalarFormatter): ScalarFormatter class from matplotlib.ticker
    """
    def _set_format(self):
        """_set_format function to format the colorbar labels to scientific notation.
        """
        self.format = "%1.3f"

class AnalysisPlotUtils:
    """AnalysisPlotUtils class to plot the attention matrix for each head and the sensitivity analysis of the model."""
    def __init__(self, arcana_procedure, sample_number):
        plots.Plots()
        self.data_headers = arcana_procedure.data_config.data_headers
        self.save_path = os.path.join(arcana_procedure.model_config.result_path,
                            "test_plots", f"sample_{sample_number}", "analysis")
        self.arch_pointer = None
        self.plot_headers = None

    def _heatmap_style(self, x_input, heat_ax, plot_title, multi_plots=False, orientation="vertical",
                    shrink_cbar=0.82, cbar_label="Attention probability"):
        """ General function to plot the attention matrix for each head.

        Args:
            x_input (_type_): _description_
            heat_ax (_type_): _description_
            plot_title (_type_): _description_
            multi_plots (bool, optional): _description_. Defaults to False.
            orientation (str, optional): _description_. Defaults to "vertical".
        """
        if multi_plots:
            cbar_kws={"shrink":shrink_cbar, "orientation":orientation,
                                "format": ticker.LogFormatterSciNotation(10, labelOnlyBase=True)}
        else:
            cbar_kws = {"format": ticker.LogFormatterSciNotation(10, labelOnlyBase=True),
                        "orientation":orientation}

        step_x_ticks = 3 if (x_input.shape[-1]+1) % 3 == 0 else 2 if (x_input.shape[-1]+1) % 2 == 0 else 1
        step_y_ticks = 3 if (x_input.shape[-2]+1) % 3 == 0 else 2 if (x_input.shape[-2]+1) % 2 == 0 else 1
        heat_plot = sns.heatmap(x_input, ax=heat_ax, cmap="viridis", square=True,
                    cbar_kws=cbar_kws,
                    linewidths=0.01,
                    #rasterized=True,
                    xticklabels=step_x_ticks, yticklabels=step_y_ticks)
        # define the font size of the labels
        heat_plot.set_yticklabels(heat_plot.get_yticklabels(), rotation=0, fontsize=3)
        heat_plot.set_xticklabels(heat_plot.get_xticklabels(), rotation=0, fontsize=3)
        # define the font size scale
        sns.set(font_scale=0.1)
        # define the colorbar font size
        heat_ax.collections[0].colorbar.ax.tick_params(labelsize=3)
        # format the color bar label to display the scientific notation
        fmt = ScalarFormatterForceFormat(useMathText=True, useOffset=False)
        fmt.set_powerlimits((0,0))

        heat_ax.collections[0].colorbar.ax.yaxis.offsetText.set_fontsize(3)
        #offset = heat_ax.collections[0].colorbar.ax.yaxis.get_offset_text()
        #offset.set_position((offset.get_position()[0], offset.get_position()[1]-0.05))

        heat_ax.collections[0].colorbar.ax.yaxis.set_major_formatter(fmt)
        # give labels to the colorbar
        heat_ax.collections[0].colorbar.set_label(cbar_label, fontsize=3, rotation=90)

        heat_ax.collections[0].colorbar.ax.tick_params(labelsize=3,
                                                    which = 'both', direction ='in',length=0.6, width=0.1)

        heat_ax.set_title(plot_title, fontsize=4)
        heat_ax.tick_params(axis='both', which='both', length=0, width=0)
        # TODO check the x_input shape for both attention types
        heat_ax.set_xticklabels(np.arange(1, x_input.shape[-1]+1,step_x_ticks), fontsize=3)
        heat_ax.set_yticklabels(np.arange(1, x_input.shape[-2]+1,step_y_ticks), fontsize=3)

    def _heatmap_save(self,file_name):
        #plt.savefig(os.path.join(self.save_path, f"{file_name}.png"))
        plt.savefig(os.path.join(self.save_path, f"{file_name}.svg"))
        plt.clf()
        plt.close()


    def _calculate_nrows_ncols(self, sensitivity):
        if len(sensitivity.keys()) < 4:
            num_rows = len(sensitivity.keys())
            num_cols = 1
        elif len(sensitivity.keys()) == 4:
            num_rows , num_cols = 2,2
        elif len(sensitivity.keys())  == 5:
            num_rows = 5
            num_cols = 1
        else:
            num_rows = 2
            num_cols = len(sensitivity.keys())//2 + len(sensitivity.keys())%2
        return num_rows, num_cols


    def _map_keys(self, variable_dict):
        # Extract last 3 keys from a
        a_keys = list(variable_dict.keys())[-3:]

        # Extract last 3 headers from self.data_headers
        variable_headers = self.data_headers[-3:]
        self.plot_headers = dict(zip(a_keys, variable_headers))


    def _plot_individual_attention_head(self, attention_probs, batch_idx=0):
        if isinstance(attention_probs, np.ndarray):
            n_heads, _, _ = attention_probs.shape
        else:
            _, n_heads, _, _ = attention_probs.shape

        n_division = 4 if n_heads < 32 else 8
        n_rows = n_heads//n_division if n_heads%n_division == 0 else n_heads//n_division + 1
        _, ax = plt.subplots(n_rows, n_division, figsize=(n_division*1.818, n_rows*1.7))

        for head_idx in range(n_heads):
            heat_ax = ax[head_idx//n_division, head_idx%n_division] if n_rows > 1 else ax[head_idx%n_division]
            if isinstance(attention_probs, np.ndarray):
                x_input = attention_probs[head_idx, :, :]

            else:
                x_input = attention_probs[batch_idx, head_idx, :, :].cpu().detach().numpy()

            self._heatmap_style(x_input=x_input, heat_ax=heat_ax,
                            plot_title= f"Head {head_idx}", multi_plots=True)

        self._heatmap_save(f"{self.arch_pointer}_attention_heads")


    def _plot_multihead_performance_on_each_sequence(self, attention_probs, batch_idx=0):
        _, ax = plt.subplots()
        if isinstance(attention_probs, np.ndarray):
            mean_attention_probs = attention_probs.mean(axis=0)
        else:
            mean_attention_probs = attention_probs.mean(dim=1)[batch_idx, :, :].cpu().detach().numpy()

        self._heatmap_style(x_input = mean_attention_probs, heat_ax=ax,
                        plot_title = "Overall performance on each sequence", multi_plots=False)
        self._heatmap_save(f"{self.arch_pointer}_overall_multihead_performance_on_each_sequence")


    def _heatmap_sensitivity(self, sensitivity, future_step, available_sequence):
        """ Plots the sensitivity analysis of the model

        Args:
            sensitivity (dict): Dictionary containing the sensitivity analysis of the model
            future_step (_type_):
            save_path (_type_): _description_
        """
        self._map_keys(sensitivity)

        for key, value in sensitivity.items():
            if isinstance(value, torch.Tensor):
                sensitivity[key] = value.cpu().detach().numpy()

        num_rows, num_cols = self._calculate_nrows_ncols(sensitivity)
        # TODO: check if normalization is necessary
        _, ax = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(num_cols*1.75, num_rows*0.9))

        for idx, (key, value) in enumerate(sensitivity.items()):
            if num_rows == 1:
                heat_ax = ax[idx%num_cols]
            elif num_cols == 1:
                heat_ax = ax[idx%num_rows]
            else:
                heat_ax = ax[idx//num_cols, idx%num_cols]
            # heat_ax = ax[idx//num_cols, idx%num_cols] if num_rows > 1 else ax[idx%num_cols]
            self._heatmap_style(x_input = value[future_step, :, :].T, heat_ax=heat_ax,
                    plot_title=f"Sensitivity for future step {future_step}", multi_plots=True,
                    shrink_cbar=0.25, cbar_label="Gradients")
            heat_ax.set_title(f"{self.plot_headers[key]}", fontsize=3)
            # Set custom y-tick labels
            num_y_ticks = value[future_step, :, :].T.shape[0]  # Number of ticks on y-axis
            custom_y_labels = self.data_headers[-num_y_ticks:]  # Slice the last 'num_y_ticks' headers
            tick_positions = np.arange(num_y_ticks) + 0.5  # Offset by half for centering

            # set y-tick and then set the y-tick labels
            heat_ax.set_yticks(tick_positions)
            heat_ax.set_yticklabels(custom_y_labels, fontsize=3, rotation=0)

            # Adjust the position of the scientific notation for the last heatmap's colorbar
            offset = heat_ax.collections[0].colorbar.ax.yaxis.get_offset_text()
            offset.set_position((offset.get_position()[0], offset.get_position()[1]+0.2))

            heat_ax.set_title(f"Analysis for {self.plot_headers[key]}", fontsize=3)
            # heat_ax.set_ylabel(f"Input dimension", fontsize=3)


        heat_ax.set_xlabel("Available sequence", fontsize=3)


        self._heatmap_save(f"{future_step + available_sequence +1}_sensitivity_analysis")


    def _line_plot_sensitivity(self, sensitivity,future_step, available_sequence, log_scale = False):
        """ Plots the sensitivity analysis of the model

        Args:
            sensitivity (_type_): _description_
            future_step (_type_): _description_
        """
        plots.Plots()
        num_rows, num_cols = self._calculate_nrows_ncols(sensitivity)
        # check if the values are tensors or numpy arrays and convert them to numpy arrays
        for key, value in sensitivity.items():
            if isinstance(value, torch.Tensor):
                sensitivity[key] = value.cpu().detach().numpy()

        # TODO: check if normalization is necessary
        _, ax = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(num_cols*3.3, num_rows*3.5))
        for idx, (key, value) in enumerate(sensitivity.items()):
            if num_rows == 1:
                plot_ax = ax[idx%num_cols]
            elif num_cols == 1:
                plot_ax = ax[idx%num_rows]
            else:
                plot_ax = ax[idx//num_cols, idx%num_cols]

            for dim_num, input_dim in enumerate(range(value.shape[2])):
                plot_ax.plot(range(1, value.shape[1]+1),value[future_step, :, input_dim].T,
                                label=f"{self.data_headers[dim_num+1]}", linewidth=0.7)

            plot_ax.set_title(f"{self.plot_headers[key]}", fontsize=5)

            plot_ax.legend(fontsize=3)
            # Enlarge x and y ticks
            plot_ax.tick_params(axis='both', labelsize=6)

            # Remove background color
            plot_ax.set_facecolor('white')
            plot_ax.figure.set_facecolor('white')

            # Move the legend to the right side, outside of the plot
            plot_ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))

            if log_scale:
                # Set the yscale to logarithmic
                plot_ax.set_yscale('log')

            plot_ax.set_ylabel("Gradient", fontsize=5)

        plot_ax.set_xlabel("Available sequence", fontsize=5)
        self._heatmap_save(f"{future_step+available_sequence+1}_{key}_sensitivity_analysis_line_plot")


    def plot_all_multihead_attention(self, attention_probs, arch_pointer="encoder", batch_idx=0):
        """ Plots the attention matrix for each head and the overall performance of the model on each sequence.

        Args:
            attention_probs (_type_): _description_
            arch_pointer (str, optional): _description_. Defaults to "encoder".
            batch_idx (int, optional): _description_. Defaults to 0.
        """
        self.arch_pointer = arch_pointer

        self._plot_individual_attention_head(attention_probs, batch_idx)
        self._plot_multihead_performance_on_each_sequence(attention_probs, batch_idx)


    def plot_additive_attention(self, attention_probs, arch_pointer):
        """ Plots the overall performance of the additive system on each sequence.

        Args:
            attention_probs (tensor or np.array): Attention probabilities
            arch_pointer (str): The architecture pointer
        """
        self.arch_pointer = arch_pointer
        _, ax = plt.subplots()
        if isinstance(attention_probs, np.ndarray):
            # check if it is one dimensional then reshape it
            if attention_probs.ndim == 1:
                attention_probs = attention_probs.reshape(1, -1)
            input_data = attention_probs
        else:
            input_data = attention_probs.cpu().detach().numpy()
        self._heatmap_style(x_input = input_data, heat_ax=ax,
                        plot_title = "Overall performance of additive system", multi_plots=False)
        self._heatmap_save(f"{self.arch_pointer}_overall_additive_performance_on_each_sequence")


    def plot_sensitivity_analyis(self, sensitivity, future_step, available_sequence, log_scale=False):
        """ Plots the sensitivity analysis of the model

        Args:
            sensitivity (tensor or np.array): Sensitivity analysis
            future_step (int): The future step
            save_path (str): The path to save the plot
        """
        self._heatmap_sensitivity(sensitivity, future_step, available_sequence)
        self._line_plot_sensitivity(sensitivity, future_step, available_sequence, log_scale)
