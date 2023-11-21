''' This module contains the functions to plot the learning rate, the train and validation loss and the prediction of the model'''
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from arcana.logger import logger
from arcana.plots import plots

plots.Plots()

log = logger.get_logger("model_helpers")


def plot_model_learning_rate(learning_rate_dict, plot_path):
    """plot the learning rate of the model

    Args:
        learning_rate_dict (dict): a dictionary containing the learning rate and epoch
        plot_path (str): the path to save the plot
    """
    _, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.scatter(learning_rate_dict['epoch'], learning_rate_dict['learning_rate'], s=3, c="#208F90")
    ax.set_ylabel("learning Rate")
    ax.set_xlabel("epoch number")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(os.path.join(plot_path, 'learning_rate.png'))
    plt.savefig(os.path.join(plot_path, 'learning_rate.svg'))
    plt.close()


def plot_train_val_loss(losses, plot_path, loss_type, train_loss_mode="batch"):
    """plot the train and validation loss for epoch or batch

    Args:
        losses (dict): a dictionary containing the train and validation loss
        plot_path (str): the path to save the plot
        loss_type (str): the type of loss to plot
        train_loss_mode (str, optional): the type of loss to plot. Defaults to "batch".
    """

    _, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.scatter(range(len(losses[f'train_loss_{train_loss_mode}'])), losses[f'train_loss_{train_loss_mode}'], s=3,
                                                                c="#DE6E4B", label="train")
    ax.scatter(range(len(losses[f'val_loss_{train_loss_mode}'])), losses[f'val_loss_{train_loss_mode}'], s=3,
                                                                c="#208F90", label="val")
    ax.set_ylabel(f"{loss_type} loss")
    ax.set_xlabel(f"{train_loss_mode} number")
    ax.legend()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(os.path.join(plot_path, f'{loss_type}_loss_{train_loss_mode}.png'))
    plt.savefig(os.path.join(plot_path, f'{loss_type}_loss_{train_loss_mode}.svg'))
    plt.close()


def plot_train_val_loss_individual(losses, plot_path, loss_type, train_loss_mode):
    """plot the train and validation loss for epoch or batch

    Args:
        losses (dict): a dictionary containing the train and validation loss
        plot_path (str): the path to save the plot
        loss_type (str): the type of loss to plot
        train_loss_mode (str, optional): the type of dimension to plot.
    """

    _, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.scatter(range(len(losses[f'train_loss_epoch_{train_loss_mode}'])),
                            losses[f'train_loss_epoch_{train_loss_mode}'], s=3, c="#DE6E4B", label=f"train  {train_loss_mode}")
    ax.scatter(range(len(losses[f'val_loss_epoch_{train_loss_mode}'])),
                    losses[f'val_loss_epoch_{train_loss_mode}'], s=3, c="#208F90", label=f"val {train_loss_mode}")


    ax.set_ylabel(f"{loss_type} loss")
    ax.set_xlabel("epoch number")
    ax.legend()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(os.path.join(plot_path, f'{loss_type}_loss_{train_loss_mode}.png'))
    plt.savefig(os.path.join(plot_path, f'{loss_type}_loss_{train_loss_mode}.svg'))
    plt.close()


#FIXME avoid these many parameters for the function
def plot_sample_prediction(original_sequence, mean_prediction, available_sequence, plot_path,
                            loss_type, scores_prediction, sample_number, upper_prediction,
                            lower_prediction, ylabels = None):
    """plot the prediction of the model

    Args:
        original_sequence (np.array): the original sequence
        mean_prediction (np.array): the mean prediction of the model
        available_sequence (int): the length of the available sequence
        plot_path (str): the path to save the plot
        loss_type (str): the type of loss to plot
        scores_prediction (dict): a dictionary containing the scores of the prediction
        uncertainty (bool): whether to plot the uncertainty or not
        random_index (int): the random index of the sequence
        upper_prediction (np.array, optional): the upper prediction of the model. Defaults to None.
        lower_prediction (np.array, optional): the lower prediction of the model. Defaults to None.
        ylabels (list, optional): the labels of the y axis. Defaults to None.
    """
    plots.Plots()
    colours = ['#208F90', '#DE6E4B']

    end_sequence_length = available_sequence+mean_prediction.shape[0]
    for dim_plot in range(mean_prediction.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

        ax.scatter(range(available_sequence, end_sequence_length),
                            mean_prediction[:, dim_plot], s=3, c=colours[0], label="Mean prediction")
        # if uncertainty:
        ax.fill_between(range(available_sequence,
                        end_sequence_length),
                        lower_prediction[:, dim_plot], upper_prediction[:, dim_plot],
                        alpha=0.4, color=colours[0], label="10-90% Percentile")

        #TODO check this and fix the dimensions
        ax.scatter(range(original_sequence[:end_sequence_length,dim_plot].shape[0]),
                    original_sequence[:end_sequence_length, dim_plot],
                                            s=3, c=colours[1], label="original sequence")
        # Create first legend with the plot labels
        first_legend = ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 0.95))
        # Add this legend manually to the current Axes.
        ax.add_artist(first_legend)

        # Create custom handles for the additional metrics
        custom_lines = [Line2D([0], [0], color='black', lw=4),
                        Line2D([0], [0], color='black', lw=4),
                        Line2D([0], [0], color='black', lw=4),
                        Line2D([0], [0], color='black', lw=4)]

        labels = [
            f"MSE: {scores_prediction[f'test_MSE_dim_{dim_plot}']:.4f}",
            f"MAE: {scores_prediction[f'test_MAE_dim_{dim_plot}']:.4f}",
            f"MAPE: {scores_prediction[f'test_MAPE_dim_{dim_plot}']:.4f}",
            f"RMSE: {scores_prediction[f'test_RMSE_dim_{dim_plot}']:.4f}"
        ]
        # Create second legend with the additional metrics
        second_legend = ax.legend(custom_lines, labels, fontsize=6, loc='upper left', bbox_to_anchor=(1.03, 0.75), handlelength=0)

        # Adjust the layout for some extra space on the right
        fig.subplots_adjust(right=0.7)
        y_label = ylabels[dim_plot+1] if ylabels else f"dim {dim_plot}"
        ax.set_ylabel(y_label)
        ax.set_xlabel("Cycles")
        #ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plt.savefig(os.path.join(plot_path, f'{sample_number}_{loss_type}_dim_{dim_plot}_prediction.png'),
        #             bbox_extra_artists=(first_legend, second_legend), bbox_inches='tight')
        plt.savefig(os.path.join(plot_path, f'{sample_number}_{loss_type}_dim_{dim_plot}_prediction.svg'),
                    bbox_extra_artists=(first_legend, second_legend), bbox_inches='tight')
        plt.close()
