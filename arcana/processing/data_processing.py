''' This module is for data preparation for the model. It includes the following functions:
    1. get_data_for_model: get the data for the model
    2. data_splits: split the data into train, validation and test data
    3. standardize_data: standardize the data based on the train data
    4. tensorized_and_pad: convert the data to tensor and pad them
    5. pad_the_splits: pad the train, validation and test data
    6. prepare_data_for_model: main functions for data preparation
'''
import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import GroupShuffleSplit
from arcana.logger import logger
from arcana.utils import utils

warnings.filterwarnings("ignore")
log = logger.get_logger("arcana.processing.data_processing")

class DataPreparation:
    """Data preparation class"""
    def __init__(self, general_config, data_config, procedure_config):
        # configurations
        self.general_config = general_config
        self.data_config = data_config
        self.procedure_config = procedure_config
        # original data
        self.df = None
        # data for the model after the splits
        self.padded_train_data = []
        self.padded_val_data = []
        self.padded_test_data = []
        # data lengths for the train, val and test data
        self.train_lengths = []
        self.val_lengths = []
        self.test_lengths = []
        # test data names for the test data after splits
        self.test_data_names = None
        # scaler for the data standardization
        self.model_data_transformation = None
        # original data splits
        self.df_train_original = None
        self.df_val_original = None
        self.df_test_original = None
        # this is just for standardization
        self.df_train_scaled = None
        self.df_val_scaled = None
        self.df_test_scaled = None
        # maximum cycles that we trained with
        self.scaled_cycle_range = None


    def get_data_for_model(self):
        """Get the data for the model"""
        original_data =\
            pd.read_csv(os.path.join(self.general_config.input_data_folder,
                            self.general_config.data_name))
        test_group = original_data["test_name"].unique().tolist()
        random_sample = random.sample(test_group, self.data_config.number_of_samples)
        data_sample = original_data[original_data["test_name"].isin(random_sample)]
        self.df = data_sample.copy()[self.data_config.data_headers]


    def data_splits(self, data, ratio):
        """Split the data into train, validation and test data"""
        splitter = GroupShuffleSplit(test_size=(1 - ratio), random_state=1)
        split_outer = splitter.split(data, groups=data["test_name"])
        split_outer_ratio = next(split_outer)
        df_first_split = data.iloc[list(split_outer_ratio[0])]
        df_second_split = data.iloc[list(split_outer_ratio[1])]

        return df_first_split, df_second_split

    def get_max_available_scaled_cycle(self):
        """Get the maximum available scaled cycle"""
        if self.procedure_config.preprocess_data:
            # NOTE: comment this is in case you want to limit the prediciton cycles to the maximum available cycles (previous training with ARCANA)
            #  max_available_cycle = self.model_data_transformation.data_max_[0]
            # Also check arcana/procedures/predicting.py line 66
            # Comment the next line out
            max_available_cycle = self.data_config.maximum_available_cycles
            min_available_cycle = self.model_data_transformation.data_min_[0]
            original_cycle_range = np.arange(min_available_cycle, max_available_cycle + 1)
            self.scaled_cycle_range = (original_cycle_range - min_available_cycle) / (max_available_cycle - min_available_cycle)
        else:
            # get the number of cycles from the self.df dataframe by filtering the test_name
            max_available_cycle = max(self.df["cycle"])
            min_available_cycle = min(self.df["cycle"])
            self.scaled_cycle_range = np.arange(min_available_cycle, max_available_cycle + 1)

        if self.procedure_config.predicting:
            if self.data_config.maximum_available_cycles > max_available_cycle:
                log.warning("The maximum available cycle is %s. The selected maximum available cycle is %s. "
                    "This might cause the model to predict the future cycles unreliably. ",
                    max_available_cycle, self.data_config.maximum_available_cycles)


    def standardize_data(self):
        """Standardize the data based on the train data"""
        # standardize the data based on the train data
        if self.procedure_config.preprocess_data:

            self.df_train_scaled, self.model_data_transformation =\
                                            utils.standardize_dataset(self.df_train_original.iloc[:, 1:])
            self.df_train_scaled.insert(0, "test_name", self.df_train_original["test_name"].values)
            # standardize the validation data based on the train data
            self.df_val_scaled = pd.DataFrame(self.model_data_transformation.transform(self.df_val_original.iloc[:, 1:]),
                                                columns=self.df_val_original.iloc[:, 1:].columns)
            self.df_val_scaled.insert(0, "test_name", self.df_val_original["test_name"].values)
            # standardize the test data based on the train data
            self.df_test_scaled = pd.DataFrame(self.model_data_transformation.transform(self.df_test_original.iloc[:, 1:]),
                                                columns=self.df_test_original.iloc[:, 1:].columns)
            self.df_test_scaled.insert(0, "test_name", self.df_test_original["test_name"].values)

        else:
            self.df_train_scaled = self.df_train_original.copy()
            self.df_val_scaled = self.df_val_original.copy()
            self.df_test_scaled = self.df_test_original.copy()


    def tensorized_and_pad(self, data, padded_data, data_lengths):
        """Convert the data to tensor and pad them

        Args:
            data (pandas dataframe): data to be converted to tensor
            padded_data (list): list of padded data
            data_lengths (list): list of data lengths

        Returns:
            padded_data (list): list of padded data
            data_lengths (list): list of data lengths
        """
        # create the padded data by grouping the data based on the test name
        for _, data_groups in data.groupby("test_name"):
            grouped_data_values = data_groups.iloc[:, 1:].values
            padded_data.append(torch.tensor(grouped_data_values))
            data_lengths.append(len(grouped_data_values))

        # convert the data to tensor and pad them
        padded_data = pad_sequence(padded_data, batch_first=True, padding_value=0)
        # create a tensor from the length of each sequence
        data_lengths = torch.tensor(data_lengths)

        return padded_data, data_lengths


    def pad_the_splits(self, train, val, test):
        """Pad the train, validation and test data

        Args:
            train (pandas dataframe): train data
            val (pandas dataframe): validation data
            test (pandas dataframe): test data

        Returns:
            padded_train (list): list of padded train data
            padded_val (list): list of padded validation data
            padded_test (list): list of padded test data
        """
        # pad the train data
        padded_train, self.train_lengths = self.tensorized_and_pad(data=train, padded_data=self.padded_train_data, data_lengths=self.train_lengths)
        # pad the validation data
        padded_val, self.val_lengths = self.tensorized_and_pad(data=val, padded_data=self.padded_val_data, data_lengths=self.val_lengths)
        # pad the test data
        padded_test, self.test_lengths = self.tensorized_and_pad(data=test, padded_data=self.padded_test_data, data_lengths=self.test_lengths)
        return padded_train, padded_val, padded_test


    def prepare_data_for_model(self):
        """Main functions for data preparation"""
        # main functions for data preparation
        self.df_train_original, second_split = self.data_splits(self.df, self.data_config.train_ratio)
        self.df_val_original, self.df_test_original = self.data_splits(second_split, self.data_config.val_test_ratio)
        self.test_data_names = self.df_test_original["test_name"].unique().tolist()
        # check if the data should be standardized
        self.standardize_data()
        self.get_max_available_scaled_cycle()
        self.padded_train_data, self.padded_val_data, self.padded_test_data =\
                self.pad_the_splits(self.df_train_scaled, self.df_val_scaled, self.df_test_scaled)


    def prepare_test_data_for_pretrained_model(self):
        """Prepare the test data for the pretrained model. This is used for the finetuning"""
        #TODO
        # load the data for testing
        self.get_data_for_model()
        # load the scaled model for transforming the test data
        if self.procedure_config.preprocess_data:
            self.model_data_transformation =\
                joblib.load(self.general_config.scaler_model)
        # get the maximum available scaled cycle
        self.get_max_available_scaled_cycle()
        self.df_test_original = self.df.copy()
        self.test_data_names = self.df_test_original["test_name"].unique().tolist()
        # standardize the test data based on the train data
        if self.procedure_config.preprocess_data:
            self.df_test_scaled =\
                pd.DataFrame(self.model_data_transformation.transform(self.df_test_original.iloc[:, 1:]),
                            columns=self.df_test_original.iloc[:, 1:].columns)
            self.df_test_scaled.insert(0, "test_name", self.df_test_original["test_name"].values)
        else:
            self.df_test_scaled = self.df_test_original.copy()
        # pad the test data
        self.padded_test_data, self.test_lengths =\
            self.tensorized_and_pad(data=self.df_test_scaled,
                                    padded_data=self.padded_test_data,
                                    data_lengths=self.test_lengths)
