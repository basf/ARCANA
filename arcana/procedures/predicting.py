''' This module is used to predict the model with the test data.'''
import os
import random
import numpy as np
import torch
from arcana.logger import logger
from arcana.utils import utils
from arcana.prediction.quantile_prediction import QuantilePredictor, metrics_helper

log = logger.get_logger("arcana.procedures.predicting")

class PredictProcedure:
    """Predicting procedure"""
    def __init__(self, arcana_procedure):
        self.arcana_procedure = arcana_procedure
        self.test_data = self.get_test_data()
        # here the pth file should be loaded
        self.pretrained_model = self.get_pretrained_model()
        # set the pretrained model to the evaluation mode
        self.pretrained_model.eval()
        self.random_indices = None
        self.available_sequences = None
        self.future_steps_predictions = None
        self.quantile_predictor = QuantilePredictor(self.arcana_procedure, self.test_data,
                                                    self.pretrained_model)

    def get_test_data(self):
        """Get the test data"""
        #TODO: check me
        if self.arcana_procedure.general_config.pretrained_model:
            if self.arcana_procedure.procedure_config.transfer_learning:
                return self.arcana_procedure.data_preparation.padded_test_data
            if self.arcana_procedure.procedure_config.predicting and (not self.arcana_procedure.procedure_config.transfer_learning):
                self.arcana_procedure.data_preparation.prepare_test_data_for_pretrained_model()
        return self.arcana_procedure.data_preparation.padded_test_data

        # if not (self.arcana_procedure.procedure_config.naive_training or \
        # 	    self.arcana_procedure.procedure_config.optuna_tuning):
        #     if not self.arcana_procedure.procedure_config.transfer_learning:
        #         self.arcana_procedure.data_preparation.prepare_test_data_for_pretrained_model()
        # return self.arcana_procedure.data_preparation.padded_test_data


    def get_pretrained_model(self):
        """Get the pretrained model"""
        #TODO:check me
        if self.arcana_procedure.general_config.pretrained_model and \
            (not self.arcana_procedure.procedure_config.transfer_learning) :
            return torch.load(self.arcana_procedure.general_config.pretrained_model)
        return self.arcana_procedure.seq_2_seq_trainer.seq2seq


    def preparation(self):
        """Prepare the data for the prediction"""
        # randomly make prediction and plot them
        self.random_indices = random.sample(list(range(len(self.test_data))),
                            self.arcana_procedure.data_config.test_sample)
        # randomly get available sequences for the prediction
        self.available_sequences = np.random.randint(
                    low=self.arcana_procedure.model_config.minimum_cycle_length,
                    high=self.arcana_procedure.model_config.maximum_cycle_length,
                    size=self.arcana_procedure.data_config.test_sample)

        # NOTE: comment this is if you want to limit the number of predictions to the number of available cycles (previous training with ARCANA)
        # Also check arcana/processing/data_processing.py line 86
        # get the number of required predictions
        # if self.arcana_procedure.data_preparation.scaled_cycle_range.shape[0] < self.arcana_procedure.data_config.maximum_available_cycles:
        #     self.arcana_procedure.data_config.maximum_available_cycles = self.arcana_procedure.data_preparation.scaled_cycle_range.shape[0]

        self.future_steps_predictions = [self.arcana_procedure.data_config.maximum_available_cycles -\
                    self.available_sequences[i] for i in range(len(self.available_sequences))]
        log.info("Start predicting with the model")


    def predicting(self, model_folder=None):
        """Predict the model

        Args:
            model_folder (str): model folder. Default is None.
        """
        if model_folder:
            self.arcana_procedure.model_config.result_path = model_folder
        # predict_quantiles(self, num_steps_to_predict, sample_number, len_available_label)
        self.preparation()
        for idx,  sample_num in enumerate(self.random_indices):
            self.quantile_predictor.predict_quantiles(
                            num_steps_to_predict=self.future_steps_predictions[idx],
                            sample_number=sample_num,
                            len_available_label=self.available_sequences[idx])
            # plot the interpretation of the predictions
            self.quantile_predictor.plot_analysis(sample_number=sample_num)
            # inverse the prediction if it is preprocessed
            self.quantile_predictor.transform_predictions_to_numpy()
            # mask out the padded values
            #self.quantile_predictor.filter_padded_zeros()
            # save the transformed predictions
            self.quantile_predictor.save_transformed_predictions()
            # calculate the loss metrics
            self.quantile_predictor.calculate_metrics()
            self.quantile_predictor.save_metrics()
            # self.quantile_predictor.save_transformed_predictions()
            # potting the prediction
            self.quantile_predictor.plot_predictions()


    def all_preparation(self):
        """Prepare the data for the prediction"""

        # randomly make prediction and plot them
        self.random_indices = random.sample(list(range(len(self.test_data))),
                            self.arcana_procedure.data_config.test_sample)
        # randomly get available sequences for the prediction
        self.available_sequences = np.random.randint(
                    low=self.arcana_procedure.model_config.minimum_cycle_length,
                    high=self.arcana_procedure.model_config.maximum_cycle_length,
                    size=self.arcana_procedure.data_config.test_sample)

        # get the number of required predictions
        self.future_steps_predictions = [self.arcana_procedure.data_config.maximum_available_cycles -\
                    self.available_sequences[i] for i in range(len(self.available_sequences))]
        log.info("Start predicting with the model")


    def predicting_all(self, fix_sequency_length, plot_prediction=True):
        """Predict the model

        Args:
            fix_sequency_length (int): the length of the available sequence
            plot_prediction (bool): whether to plot the prediction or not
        """
        all_original_data, all_target_data, all_predictions = [], [], []
        self.quantile_predictor.metrics_preparation()
        if self.arcana_procedure.data_preparation.scaled_cycle_range.shape[0] < self.arcana_procedure.data_config.maximum_available_cycles:
            self.arcana_procedure.data_config.maximum_available_cycles = self.arcana_procedure.data_preparation.scaled_cycle_range.shape[0]
        future_steps_predictions =\
            self.arcana_procedure.data_config.maximum_available_cycles - fix_sequency_length
        for _,  sample_num in enumerate(range(len(self.test_data))):
            self.quantile_predictor.predict_quantiles(
                            num_steps_to_predict=future_steps_predictions,
                            sample_number=sample_num,
                            len_available_label=fix_sequency_length)
            # inverse the prediction if it is preprocessed
            self.quantile_predictor.transform_predictions_to_numpy()
            self.quantile_predictor.save_transformed_predictions()
            all_original_data.append(self.quantile_predictor.complete_sample_sequence)
            all_target_data.append(self.quantile_predictor.target_labels)
            all_predictions.append(self.quantile_predictor.all_transformed_predictions[1])
            if plot_prediction:
                # potting the prediction
                self.quantile_predictor.calculate_metrics()
                #self.quantile_predictor.plot_analysis(sample_number=sample_num)
                self.quantile_predictor.plot_predictions()

        # align and truncate the data
        truncated_all_predictions, truncated_all_targets =\
            utils.align_and_truncate_samples(all_predictions=np.array(all_predictions), all_target_data_list=all_target_data)

        # individual metrics computation
        for data_dim in range(self.quantile_predictor.input_size):
            self.quantile_predictor.scores_metrics[f"test_MSE_dim_{data_dim}"],\
                self.quantile_predictor.scores_metrics[f"test_RMSE_dim_{data_dim}"],\
                    self.quantile_predictor.scores_metrics[f"test_MAPE_dim_{data_dim}"],\
                        self.quantile_predictor.scores_metrics[f"test_MAE_dim_{data_dim}"] = \
                    metrics_helper(target_labels=truncated_all_targets[:, data_dim: data_dim + 1],
                                        predict_labels=truncated_all_predictions[:, data_dim: data_dim + 1])

        self.quantile_predictor.result_path =\
            os.path.join(self.arcana_procedure.model_config.result_path, "model_interpretation")
        self.quantile_predictor.sample_number = "all"
        self.quantile_predictor.save_metrics()
