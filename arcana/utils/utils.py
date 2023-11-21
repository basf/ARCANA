"""Helper functions for the arcana package."""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from arcana.logger import logger


log = logger.get_logger("arcana.helpers")


def create_dir(directory):
    """Checks if a directory is present, if not creates one at the given location
    Args:
        directory (str): Location where the directory should be created
    Returns:
        str: Location of the directory
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
        log.debug(f"Created directory {directory}.")
    else:
        log.debug(f"Directory {directory} already exists.")
    return directory


def save_plots(path, name: str = None):
    """Save plots to a directory

    Args:
        path (str): path to the directory
        name (str, optional): name of the plot. Defaults to None.
    """
    plot_dir = create_dir(path)
    plt.savefig(os.path.join(plot_dir, f"{name}.png"))
    plt.savefig(os.path.join(plot_dir, f"{name}.svg"))
    plt.clf()


def standardize_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize data

    Args:
        data (pd.DataFrame): dataframe with the data

    Returns:
        scaled_data (pd.DataFrame): dataframe with the scaled data
        scaler (sklearn.preprocessing.MinMaxScaler): scaler used to scale the data
    """
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return scaled_data, scaler


def prepare_folder_structure(test_id):
    """Prepare the folder structure for the results

    Args:
        test_id (str): ID of the test
    """
    result_path = create_dir(os.path.join("results", test_id))
    temp_vis_path = create_dir(os.path.join(result_path, "temp_models"))
    create_dir(os.path.join(result_path, "train_parameters"))
    create_dir(os.path.join(temp_vis_path, "temp_train_vis"))
    create_dir(os.path.join(result_path, "test_plots"))
    create_dir(os.path.join(result_path, "model_interpretation"))
    create_dir(os.path.join(result_path, "data_splits"))
    create_dir(os.path.join(result_path, "model_parameters"))
    create_dir(os.path.join(result_path, "config_files"))

    return result_path


def handle_tensor(obj):
    """Handle the tensor objects

    Args:
        obj (torch.Tensor): tensor object
    """
    if torch.is_tensor(obj):
        return obj.tolist()  # Convert tensor to list
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def prepare_optuna_folder_structure(trial_path):
    """Prepare the folder structure for the results

    Args:
        test_id (str): ID of the test
    """
    temp_vis_path = create_dir(os.path.join(trial_path, "temp_models"))
    create_dir(os.path.join(trial_path, "train_parameters"))
    create_dir(os.path.join(temp_vis_path, "temp_train_vis"))
    create_dir(os.path.join(trial_path, "test_plots"))
    create_dir(os.path.join(trial_path, "model_interpretation"))
    create_dir(os.path.join(trial_path, "model_parameters"))
    create_dir(os.path.join(trial_path, "config_files"))

def save_optuna_fig(save_path, plot_type):
    """Save the figure

    Args:
        save_path (str): path to the directory
        plot_type (str): type of the plot
    """
    plt.savefig(f"{save_path}/{plot_type}.png")
    plt.savefig(f"{save_path}/{plot_type}.svg")
    plt.close()


def save_test_data(model, model_folder, test_data, test_lengths):
    """Save the test data and the test lengths

    Args:
        model (torch.nn.Module): the model
        model_folder (str): the path to the model folder
        test_data (torch.Tensor): the test data
        test_lengths (torch.Tensor): the test lengths
    """
    np.save(os.path.join(model_folder, "model_parameters", f'test_data_{model.model_name}.npy'), test_data.cpu())
    np.save(os.path.join(model_folder, "model_parameters", f'test_lengths_{model.model_name}.npy'), test_lengths.cpu())


def pad_array_to_length(arr, target_length):
    """Pads an array with NaN values up to the target length."""
    pad_length = target_length - len(arr)
    if pad_length > 0:
        return np.concatenate([arr, np.full(pad_length, np.nan)])
    return arr


def align_and_truncate_samples(all_predictions, all_target_data_list):
    """Align and truncate the samples in the array of predictions and list of targets.

    Args:
        all_predictions (np.ndarray): Array of predictions
        all_target_data_list (list): List of targets

    Returns:
        truncated_all_predictions (np.ndarray): Truncated array of predictions
        truncated_all_targets (np.ndarray): Truncated array of targets
    """
    truncated_preds_list = []
    truncated_targets_list = []

    for pred, target in zip(all_predictions, all_target_data_list):
        min_length = min(pred.shape[0], target.shape[0])
        truncated_pred = pred[:min_length, :]
        truncated_target = target[:min_length, :]

        truncated_preds_list.append(truncated_pred)
        truncated_targets_list.append(truncated_target)

    # Concatenate along the time axis (axis=0)
    truncated_all_predictions = np.concatenate(truncated_preds_list, axis=0)
    truncated_all_targets = np.concatenate(truncated_targets_list, axis=0)

    return truncated_all_predictions, truncated_all_targets
