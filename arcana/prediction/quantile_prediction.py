''' This module contains the class that predicts with the quantile model'''
import os
import json
import copy
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error

from arcana.logger import logger
from arcana.utils import utils
from arcana.plots import plot_utils
from arcana.plots.analysis_plot_utils import AnalysisPlotUtils
from arcana.models.encoders.additive_encoder import AdditiveEncoder
from arcana.models.encoders.multihead_encoder import MultiheadEncoder


log = logger.get_logger("arcana.prediction.quantile_prediction")

class QuantilePredictor:
    """Predicting with the quantile model"""
    def __init__(self, arcana_procedure, test_data, pretrained_model):
        self.arcana_procedure = arcana_procedure
        self.test_data = test_data
        self.all_sequence_input = None
        self.test_input = None
        self.test_length = None
        self.pretrained_model = pretrained_model
        self.attention_probs_encoder = None
        self.all_attention_samples = None
        self.all_sensitivity_grad = None

        self.output_size = self.arcana_procedure.model_config.output_size
        self.dim_depth = self.pretrained_model.dim_depth
        self.input_size = self.arcana_procedure.model_config.input_size
        self.result_path = None
        self.prediction_plot_path = None
        self.analysis_plot_path = None
        self.prediction_path = None

        self.num_steps_to_predict = None
        self.sample_number = None
        self.len_available_label = None

        self.complete_sample_sequence = None
        self.target_labels = None

        self.predictions_qunatile_1 = None
        self.predictions_qunatile_2 = None
        self.predictions_qunatile_3 = None

        self.all_quantile_predictions = None
        self.all_transformed_predictions = None

        self.scores_metrics = None
        self.analysis_utils = None



    def predict_data_preparation(self, num_steps_to_predict, sample_number, len_available_label):
        """Prepare the data for prediction

        Args:
            num_steps_to_predict (int): number of steps to predict
            sample_number (int): sample number
            len_available_label (int): length of the available sequence
        """
        log.info(f"Preparing data for prediction for sample {sample_number}")

        self.result_path = utils.create_dir(os.path.join(self.arcana_procedure.model_config.result_path,
                                    "model_interpretation", f"sample_{sample_number}", "analysis"))
        self.prediction_path = utils.create_dir(os.path.join(self.arcana_procedure.model_config.result_path,
                                    "model_interpretation", f"sample_{sample_number}", "predictions"))
        self.analysis_plot_path = utils.create_dir(os.path.join(self.arcana_procedure.model_config.result_path,
                                    "test_plots", f"sample_{sample_number}", "analysis"))
        self.prediction_plot_path  = utils.create_dir(os.path.join(self.arcana_procedure.model_config.result_path,
                                    "test_plots", f"sample_{sample_number}", "predictions"))



        self.num_steps_to_predict = num_steps_to_predict
        self.sample_number = sample_number
        self.len_available_label = len_available_label

        self.complete_sample_sequence =\
            self.test_data[self.sample_number:self.sample_number+1, :len(self.arcana_procedure.data_preparation.scaled_cycle_range)-1, :].detach().clone()

        if (self.complete_sample_sequence.shape[1] < self.arcana_procedure.data_config.maximum_available_cycles):
            self.complete_sample_sequence = torch.cat((self.complete_sample_sequence,
                torch.zeros(1, self.arcana_procedure.data_config.maximum_available_cycles - self.complete_sample_sequence.shape[1],
                self.complete_sample_sequence.shape[2])), dim=1)
        self.apply_correct_exogenous()

        self.all_sequence_input = self.complete_sample_sequence[:, :self.len_available_label, :].clone()
            # self.test_data[self.sample_number:self.sample_number+1,:self.len_available_label, :]
        self.test_input = torch.tensor(self.all_sequence_input,
                            requires_grad=True).to(self.arcana_procedure.device)
        self.test_length = torch.tensor([len(test) for test in self.test_input]).type(torch.int32)
        self.all_sensitivity_grad = {
                f"sensitivity_dimension_{dim}": torch.zeros(self.num_steps_to_predict,
                self.len_available_label, self.all_sequence_input.shape[2]) \
                for dim in range(self.output_size)}

        number_of_samples = self.all_sequence_input.shape[0]
        if self.all_sequence_input.shape[2] == self.output_size:
            self.predictions_qunatile_2 = torch.zeros(number_of_samples, self.num_steps_to_predict, self.output_size)
            self.predictions_qunatile_1 = torch.zeros(number_of_samples, self.num_steps_to_predict, self.output_size)
            self.predictions_qunatile_3 = torch.zeros(number_of_samples, self.num_steps_to_predict, self.output_size)

        else:
            self.predictions_qunatile_2 = torch.zeros(number_of_samples, self.num_steps_to_predict, self.output_size+self.dim_depth)
            self.predictions_qunatile_1 = torch.zeros(number_of_samples, self.num_steps_to_predict, self.output_size+self.dim_depth)
            self.predictions_qunatile_3 = torch.zeros(number_of_samples, self.num_steps_to_predict, self.output_size+self.dim_depth)

        self.all_quantile_predictions = None
        self.all_transformed_predictions = None


    def metrics_preparation(self):
        """Prepare the metrics for prediction"""
        log.info(f"Preparing metrics for prediction for sample {self.sample_number}")

        self.scores_metrics = {"test_MSE": None, "test_RMSE": None, "test_MAPE": None, "test_MAE": None}
        score_metrics_individual = {f"test_MSE_dim_{i}" : None for i in range(self.input_size)}
        score_metrics_individual.update({f"test_RMSE_dim_{i}" : None for i in range(self.input_size)})
        score_metrics_individual.update({f"test_MAPE_dim_{i}" : None for i in range(self.input_size)})
        score_metrics_individual.update({f"test_MAE_dim_{i}" : None for i in range(self.input_size)})
        self.scores_metrics.update(score_metrics_individual)


    def apply_correct_exogenous(self):
        """Apply the correct exogenous data"""
        zero_indices = torch.all(self.complete_sample_sequence.reshape(-1, self.complete_sample_sequence.shape[-1]) == 0, dim=1)
        if torch.any(zero_indices):
            # Get the len of non-zero cycle
            len_non_zero_cycle = len(self.complete_sample_sequence.reshape(-1, self.complete_sample_sequence.shape[-1])[~zero_indices, 0])
            # Replace zeros in the first dimension (cycle) with the corresponding values from the cycle range
            self.complete_sample_sequence.reshape(-1, self.complete_sample_sequence.shape[-1])[zero_indices, 0] =\
                torch.tensor(self.arcana_procedure.data_preparation.scaled_cycle_range[
                    len_non_zero_cycle:len_non_zero_cycle+torch.sum(zero_indices)],dtype=torch.float64)
            for depth in range(1, self.dim_depth):
                # get the median of the past values with window length
                past_values =\
                    self.complete_sample_sequence.reshape(-1, self.complete_sample_sequence.shape[-1])[~zero_indices, depth][-10:]
                median_past_values = torch.median(past_values)
                # Replace zeros in the other dimensions (depth) with the corresponding values from the previous depth
                self.complete_sample_sequence.reshape(-1, self.complete_sample_sequence.shape[-1])[zero_indices, depth] =\
                    median_past_values



    def predict_quantiles(self, num_steps_to_predict, sample_number, len_available_label):
        """Predict the quantiles. This function also saves the attention weights and sensitivity scores

        Args:
            num_steps_to_predict (int): number of steps to predict
            sample_number (int): sample number
            len_available_label (int): length of the available sequence
        """
        self.predict_data_preparation(num_steps_to_predict, sample_number, len_available_label)

        log.info(f"Predicting quantiles for sample {sample_number}")

        torch.backends.cudnn.enabled=False
        # put the model in evaluation mode
        self.pretrained_model.eval()
        # get the encoder outputs
        if isinstance(self.pretrained_model.encoder, AdditiveEncoder):
            encoder_outputs, hidden, cell = self.pretrained_model.encoder(self.test_input, self.test_length)
            self.all_attention_samples = {f"future_step_{step}": torch.zeros(1,
                        self.len_available_label) for step in range(self.num_steps_to_predict)}
        elif isinstance(self.pretrained_model.encoder, MultiheadEncoder):
            encoder_outputs, hidden, cell, self.attention_probs_encoder =\
                        self.pretrained_model.encoder(self.test_input, self.test_length)
            self.all_attention_samples = {f"future_step_{step}":
                            torch.zeros(1, self.arcana_procedure.model_config.nhead_decoder,
                                        1,self.len_available_label) for step in range(self.num_steps_to_predict)}

        # create the input of the decoder
        decoder_x = self.test_input[:, -self.pretrained_model.window_length:, :].clone()

        # run the model for the future steps
        for future_step in range(self.num_steps_to_predict):
            # Clear the gradient at the beginning of each step
            self.pretrained_model.zero_grad()


            # Get the prediction from the decoder
            prediction, hidden, cell, attention_probs_decoder =\
                        self.pretrained_model.decoder(decoder_x, hidden, cell, encoder_outputs)
            pred_qunatile_1, pred_qunatile_2, pred_qunatile_3 = prediction


            # get the attention from the decoder
            if isinstance(self.pretrained_model.encoder, AdditiveEncoder):
                self.all_attention_samples[f"future_step_{future_step}"][:, :] =\
                                                      attention_probs_decoder.detach().clone()
            elif isinstance(self.pretrained_model.encoder, MultiheadEncoder):
                self.all_attention_samples[f"future_step_{future_step}"][:, :, :, :] =\
                                                      attention_probs_decoder.detach().clone()

            # get the sensitivity from the decoder
            # Calculate gradient for each output dimention
            for output_dim in range(self.output_size):
                if self.test_input.grad is not None:
                    self.test_input.grad.zero_()

                pred_qunatile_2[:, :, output_dim].sum().backward(inputs = self.test_input, retain_graph=True)
                # prediction[:, :, output_dim].sum().backward(retain_graph=True)
                  # grad(outputs = prediction[:, :, output_dim].sum(), inputs = test_input, creategraph = True, retain_graph=True)[0]
                sensitivity_score = self.test_input.grad.abs().detach()
                self.all_sensitivity_grad[f"sensitivity_dimension_{output_dim}"][future_step, :, :] = sensitivity_score.clone()


            # append predicted values and the quantiles
            pred_qunatile_2 = torch.cat((self.complete_sample_sequence[:,
                                        self.len_available_label+future_step, 0:self.dim_depth].unsqueeze(1),
                                        pred_qunatile_2), dim=2)
            pred_qunatile_1 = torch.cat((self.complete_sample_sequence[:,
                                        self.len_available_label+future_step, 0:self.dim_depth].unsqueeze(1),
                                        pred_qunatile_1), dim=2)
            pred_qunatile_3 = torch.cat((self.complete_sample_sequence[:,
                                        self.len_available_label+future_step, 0:self.dim_depth].unsqueeze(1),
                                        pred_qunatile_3), dim=2)


            # save the prediction
            self.predictions_qunatile_1[:, future_step, :] = pred_qunatile_1.detach().clone()
            self.predictions_qunatile_3[:, future_step, :] = pred_qunatile_3.detach().clone()
            self.predictions_qunatile_2[:, future_step, :] = pred_qunatile_2.detach().clone()
            # update the decoder input
            decoder_x = torch.cat((decoder_x[:, 1:, :], pred_qunatile_2), dim=1)

        # concat all the predictions and their quantiles
        self.all_quantile_predictions = (self.predictions_qunatile_1, self.predictions_qunatile_2, self.predictions_qunatile_3)
        # call saving functions
        self.save_attention()
        self.save_sensitivity()
        self.save_predictions()
        torch.backends.cudnn.enabled=True


    def save_attention(self):
        """Save the attention weights"""
        log.info(f"Saving attention weights for sample {self.sample_number}")
        torch.save(self.all_attention_samples,
                   os.path.join(self.result_path,
                    f"attention_weights_sample_{self.sample_number}_available_{self.len_available_label}_cycles.pt"))

    def save_sensitivity(self):
        """Save the sensitivity scores"""
        log.info(f"Saving sensitivity scores for sample {self.sample_number}")
        torch.save(self.all_sensitivity_grad,
                   os.path.join(self.result_path,
                    f"sensitivity_grad_sample_{self.sample_number}_available_{self.len_available_label}_cycles.pt"))

    def save_predictions(self):
        """Save the predictions"""
        log.info(f"Saving predictions for sample {self.sample_number}")
        # save the predictions
        prefix = f"sample_{self.sample_number}_available_{self.len_available_label}_cycles_{self.arcana_procedure.procedure_config.attention_type}"
        np.save(os.path.join(self.prediction_path, f"predictions_{prefix}.npy"), self.predictions_qunatile_2.cpu().numpy())
        np.save(os.path.join(self.prediction_path, f"predictions_upper_quantile_{prefix}.npy"), self.predictions_qunatile_1.cpu().numpy())
        np.save(os.path.join(self.prediction_path, f"predictions_lower_quantile_{prefix}.npy"), self.predictions_qunatile_3.cpu().numpy())


    def plot_analysis(self, sample_number):
        """Plot the analysis of the prediction

        Args:
            sample_number (int): sample number

        """
        self.analysis_utils = AnalysisPlotUtils(arcana_procedure=self.arcana_procedure, sample_number=sample_number)
        log.info(f"Plotting analysis, attention and sensitivity for sample {self.sample_number}")
        if isinstance(self.pretrained_model.encoder, MultiheadEncoder):
            self.analysis_utils.plot_all_multihead_attention(attention_probs=self.attention_probs_encoder,
                                                            arch_pointer="encoder")

        for step in range(self.num_steps_to_predict):
            #if (step % 15 == 0) and (self.sample_number % 10 == 0):
            if step % 100 == 0:
                self.analysis_utils.plot_sensitivity_analyis(
                    sensitivity=self.all_sensitivity_grad, future_step=step,
                    available_sequence=self.len_available_label, log_scale=True)

                attentions_probability = self.all_attention_samples[f"future_step_{step}"]
                pointer = f"decoder_{self.len_available_label + step + 1}"
                # plot the attention weights
                if isinstance(self.pretrained_model.encoder, AdditiveEncoder):
                    self.analysis_utils.plot_additive_attention(
                        attention_probs= attentions_probability,
                        arch_pointer=pointer)
                elif isinstance(self.pretrained_model.encoder, MultiheadEncoder):
                    self.analysis_utils.plot_all_multihead_attention(
                        attention_probs=self.all_attention_samples[f"future_step_{step}"],
                        arch_pointer=pointer)


    def transform_predictions_to_numpy(self):
        """Helper function that transform the predictions to numpy and to original scale"""

        log.info(f"Transforming predictions to numpy and to original scale for sample {self.sample_number}")
        self.all_transformed_predictions = []
        # create a mask for the complete sample sequence to remove the padded zeros
        mask = (self.complete_sample_sequence[:,:,self.dim_depth:] != 0).any(dim=2)
        masked_sample = self.complete_sample_sequence[mask]

        if self.arcana_procedure.procedure_config.preprocess_data:
            for prediction in self.all_quantile_predictions:
                scaled_prediction =\
                    self.arcana_procedure.data_preparation.model_data_transformation.inverse_transform(
                        prediction.cpu().numpy().squeeze(0))
                self.all_transformed_predictions.append(scaled_prediction)

            self.complete_sample_sequence =\
                    self.arcana_procedure.data_preparation.model_data_transformation.inverse_transform(
                                                masked_sample.cpu().numpy())
        else:
            for prediction in self.all_quantile_predictions:
                self.all_transformed_predictions.append(prediction.cpu().numpy().squeeze(0))

            self.complete_sample_sequence = masked_sample.cpu().numpy()

        self.target_labels = copy.deepcopy(self.complete_sample_sequence[self.len_available_label:, :])


    def save_transformed_predictions(self):
        """ Helper function that saves the transformed predictions"""
        log.info(f"Saving transformed predictions for sample {self.sample_number}")
        # save the predictions
        np.save(os.path.join(self.prediction_path, f"original_predictions_{self.sample_number}.npy"),
                    np.array(self.all_transformed_predictions))
        np.save(os.path.join(self.prediction_path, f"original_sequence_{self.sample_number}.npy"),
                    self.complete_sample_sequence)
        self._data_frame_of_predictions()


    def calculate_metrics(self):
        """Calculate the metrics of the predictions"""
        self.metrics_preparation()
        log.info(f"Calculating metrics for prediction for sample {self.sample_number}")
        predict_labels = self.all_transformed_predictions[1]

        min_length = min(self.target_labels.shape[0], predict_labels.shape[0])
        # overall metrics computation
        self.scores_metrics["test_MSE"], self.scores_metrics["test_RMSE"],self.scores_metrics["test_MAPE"], self.scores_metrics["test_MAE"] = \
            metrics_helper(target_labels=self.target_labels[:min_length, :], predict_labels=predict_labels[:min_length, :])

        # individual metrics computation
        for data_dim in range(self.input_size):
            self.scores_metrics[f"test_MSE_dim_{data_dim}"], self.scores_metrics[f"test_RMSE_dim_{data_dim}"], \
            self.scores_metrics[f"test_MAPE_dim_{data_dim}"], self.scores_metrics[f"test_MAE_dim_{data_dim}"] = \
                    metrics_helper(target_labels=self.target_labels[:min_length, data_dim: data_dim + 1],
                                        predict_labels=predict_labels[:min_length, data_dim: data_dim + 1])

    def save_metrics(self):
        """Helper function that saves the metrics"""
        log.info(f"Saving metrics of the predictions and targets to json for sample {self.sample_number}")
        # save the scores as a dictionary
        with open(os.path.join(self.result_path, f"scores_{self.sample_number}.json"), "w", encoding='utf-8') as f:
            json.dump(self.scores_metrics, f)


    def _data_frame_of_predictions(self):
        """Helper function that creates a data frame of predictions and targets."""
        log.info(f"Creating DataFrame of predictions and targets for sample {self.sample_number}")

        # Initialize an empty dictionary to hold data
        data = {}

        # List of keys and data for different types of predictions and sequences
        keys_and_data_list = [
            ("prediction_dim_", self.all_transformed_predictions[1]),
            ("original_sequence_dim_", self.target_labels),
            ("lower_prediction_dim_", self.all_transformed_predictions[2]),
            ("upper_prediction_dim_", self.all_transformed_predictions[0])
        ]

        # Find the maximum length across all arrays
        max_length = max(
            len(arr[:, 0].flatten()) for key, arr in keys_and_data_list
        )

        # Iterate through keys and data to pad and add to dictionary
        for key_prefix, arr in keys_and_data_list:
            for i in range(self.input_size):
                array_slice = arr[:, i].flatten()
                padded_array = utils.pad_array_to_length(array_slice, max_length)
                data.update({f"{key_prefix}{i}": padded_array})

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.prediction_path, f"prediction_and_targets_original_{self.sample_number}.csv"), index=False)


    def plot_predictions(self):
        """Helper function that plots the predictions"""
        log.info(f"Plotting predictions for sample {self.sample_number}")
        # plot the predictions
        plot_utils.plot_sample_prediction(
            original_sequence=self.complete_sample_sequence,
            mean_prediction=self.all_transformed_predictions[1],
            available_sequence=self.len_available_label,
            plot_path=self.prediction_plot_path,
            loss_type=self.arcana_procedure.model_config.loss_type,
            scores_prediction=self.scores_metrics,
            sample_number=self.sample_number,
            upper_prediction=self.all_transformed_predictions[0],
            lower_prediction=self.all_transformed_predictions[2],
            ylabels=self.arcana_procedure.data_config.data_headers)


def metrics_helper(target_labels, predict_labels):
    """Helper function that calculates the metrics

    Args:
        target_labels (numpy array): target labels
        predict_labels (numpy array): predicted labels

    Returns:
        mse_score (float): mean squared error
        rmse_score (float): root mean squared error
        mape_score (float): mean absolute percentage error
        mae_score (float): mean absolute error
    """
    # MSE calculation
    mse_score = mean_squared_error(target_labels, predict_labels)
    # RMSE calculation
    rmse_score = np.sqrt(mse_score)
    # MAPE calculation
    mape_score = 100 * np.mean(np.abs((target_labels - predict_labels) / target_labels))
    # MAE calculation
    mae_score = mean_absolute_error(target_labels, predict_labels)
    return mse_score, rmse_score, mape_score, mae_score
