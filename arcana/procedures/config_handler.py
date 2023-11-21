''' This module contains the ConfigHandler class which is used to read the config files and set the config dataclasses.'''
import os
import json
import shutil
import configparser
from dataclasses import dataclass

from arcana.utils import utils

class ModelConfig:
    """Model config dataclass"""
    def __init__(self):
        # model_settings
        self.input_size = None
        self.output_size = None
        self.loss_type = None
        self.minimum_cycle_length = None
        self.maximum_cycle_length = None
        self.dim_weights = None
        # optimizer
        self.learning_rate = None
        self.weight_decay = None
        # scheduler_reduced
        self.factor_reduced = None
        # scheduler_cycle
        self.step_size_up = None
        # model_parameters
        self.number_of_epochs = None
        self.hidden_dim = None
        self.batch_size = None
        self.bidirectional = None
        self.output_dropout = None
        self.window_length = None
        self.warmup_steps = None
        # encoder
        self.dropout_encoder = None
        self.num_layers_encoder = None
        # decoder
        self.dropout_decoder = None
        self.num_layers_decoder = None
        # multihead_attention
        self.nhead_encoder = None
        self.nhead_decoder = None
        # early_stopping
        self.early_stopping_type = None
        self.early_stopping_alpha = None
        self.patience= None
        # teacher_forcing
        self.tl_start_ratio = None
        self.tl_end_ratio = None
        self.epoch_division  = None
        self.decay_stride = None
        self.path_to_config = "config/model_parameter.ini"
        self.path_to_tuning_config= None
        self.result_path= None
        # huber_loss
        self.beta = None
        self.reduction = None
        # quantile_loss
        self.delta = None
        # number of trials
        self.number_of_trials = None

    def read_tuning_conf(self, trial):
        """Read the tuning config file and set the model config attributes to the values in the config file
        Args:
            trial (optuna.trial.Trial): Trial object
        """
        config = configparser.ConfigParser()
        config.read(self.path_to_tuning_config)

        for section in config.sections():
            for key, value in config[section].items():
                # Handle inline comments
                value = value.split(';')[0].strip()
                setattr(self, key, self._parse_value(trial, key, value))

    def _parse_value(self, trial, key, value):
        # If it's a list
        if value.startswith("["):
            value_list = self._get_tuning_config_list(value)

            # If the list has bool values
            if all(isinstance(val, bool) for val in value_list):
                return trial.suggest_categorical(key, value_list)
            # If the list has float values
            if any(isinstance(val, float) for val in value_list):
                # If the list is 3 items long, use [min, max, step]
                if len(value_list) == 3:
                    return trial.suggest_float(key, value_list[0], value_list[1], step=value_list[2])
                return trial.suggest_float(key, min(value_list), max(value_list))
            # If the list has int values
            if all(isinstance(val, int) for val in value_list):
                # If the list is 3 items long, use [min, max, step]
                if len(value_list) == 3:
                    return trial.suggest_int(key, value_list[0], value_list[1], step=value_list[2])
                return trial.suggest_int(key, min(value_list), max(value_list))
            raise TypeError("The type of the list is not supported")
        # If it's a bool value
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"
        # If it's a float value
        if "." in value:
            return float(value)
        # If it's an int value
        return int(value)

    def _get_tuning_config_list(self, value):
        # Assuming the values are comma separated and enclosed in []
        value = value[1:-1].strip()  # Removing []
        value_list = value.split(',')
        parsed_list = []
        for val in value_list:
            val = val.strip()
            if val.lower() in ["true", "false"]:
                parsed_list.append(val.lower() == "true")
            elif ("." in val) or ('e-' in val.lower()) or ('e+' in val.lower()):
                parsed_list.append(float(val))
            else:
                parsed_list.append(int(val))
        return parsed_list


@dataclass
class DataConfig:
    """Data config dataclass"""
    data_headers: list = None
    number_of_samples: int = None
    test_sample: int = None
    maximum_available_cycles: int = None
    train_ratio: float = None
    val_test_ratio: float = None
    path_to_config: str = "config/general_parameter.ini"

@dataclass
class GeneralConfig:
    """General config dataclass"""
    test_id: str = None
    input_data_folder: str = None
    data_name : str = None
    pretrained_model: str = None
    scaler_model: str = None
    path_to_config: str = "config/general_parameter.ini"

@dataclass
class ProcedureConfig:
    """Procedure config dataclass"""
    naive_training : bool = None
    predicting: bool = None
    preprocess_data: bool = None
    optuna_tuning: bool = None
    learning_rate_type: str = None
    attention_type: str = None
    number_of_trials: int = None
    transfer_learning: bool = None
    path_to_config: str = "config/general_parameter.ini"

class ConfigHandler:
    """Config handler class"""
    def __init__(self):
        """Initialize the config handler by initializing all the config dataclasses
            and setting the config values
        """
        self.general_config = GeneralConfig()
        self.data_config = DataConfig()
        self.procedure_config = ProcedureConfig()
        self.model_config = ModelConfig()

        self._read_general_config()
        self._read_data_config()
        self._read_procedure_config()
        self._read_model_config()
        self.model_config.result_path = utils.prepare_folder_structure(self.general_config.test_id)
        self._backup_config(self.model_config.result_path)

    def _backup_config(self, result_path):
        """Backup the config files

        Args:
            result_path (str): result path
        """
        config_path = os.path.join(result_path, "config_files")
        shutil.copy2(self.general_config.path_to_config, config_path)
        if self.procedure_config.optuna_tuning:
            shutil.copy2(self.model_config.path_to_tuning_config, config_path)
        else:
            shutil.copy2(self.model_config.path_to_config, config_path)

    def _read_general_config(self):
        """Set the general config"""
        config = configparser.ConfigParser()
        config.read(self.general_config.path_to_config)
        config.sections()
        config = config['general']
        self.general_config.test_id = self._get_config_string(config["test_id"])
        self.general_config.input_data_folder = self._get_config_string(config["input_data_folder"])
        self.general_config.data_name = self._get_config_string(config["data_name"])
        self.general_config.pretrained_model = self._get_config_string(config["pretrained_model"])
        self.general_config.scaler_model = self._get_config_string(config["scaler_model"])

    def get_general_config(self):
        """Get the general config

        Returns:
            GeneralConfig: general config
        """
        return self.general_config


    def _read_data_config(self):
        config = configparser.ConfigParser()
        config.read(self.general_config.path_to_config)
        config.sections()
        config = config['data']
        self._parse_config_section(self.data_config, config)

    def get_data_config(self):
        """Get the data config

        Returns:
            DataConfig: data config
        """
        return self.data_config


    def _read_procedure_config(self):
        """Set the procedure config"""
        config = configparser.ConfigParser()
        config.read(self.general_config.path_to_config)
        config.sections()
        config = config['procedure']
        self._parse_config_section(self.procedure_config, config)

        if self.procedure_config.naive_training and self.procedure_config.optuna_tuning:
            raise ValueError("Naive training and optuna tuning cannot be run at the same time."\
                            "Please set one of them to False in the general_parameter.ini file"\
                            "and run the program again.")

    def get_procedure_config(self):
        """Get the procedure config

        Returns:
            ProcedureConfig: procedure config
        """
        return self.procedure_config


    def _read_model_config(self):
        config = configparser.ConfigParser()
        config.read(self.model_config.path_to_config)
        # Loop through each section in the config
        for section in config.sections():
            self._parse_config_section(self.model_config, config[section])

    def get_model_config(self):
        """Get the model config

        Returns:
            ModelConfig: model config
        """
        return self.model_config

    def _parse_config_section(self, config_class, config_sec):
        """Parse the config section by determining the type of the config and assigning the value
        to the corresponding attribute in the dataclass. This works for simple strings, bools, ints, floats and lists.
        It does not work strings with "." in them, e.g. "1.0.0" or a path to a file. like "data/test.csv"
        for this cases the _get_config_string and _get_config_list methods are used.
        Args:
            config_class (class): config class
            config_sec (dict): config section
        """
        for key, value in config_sec.items():
            # Handle inline comments
            value = value.split(';')[0].strip()
            # Convert certain types from string
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif ('.' in value and (not value.startswith("[")) and ('\\' not in value) and ('/' not in value)):
                value = float(value)
            elif value.isdigit():
                value = int(value)
            elif value.startswith("["):
                value = self._get_config_list(value)
            elif ('e-' in value.lower()) or ('e+' in value.lower()):
                value = float(value)
            else:
                # Remove extra quotes if present
                value = self._get_config_string(value)
            # Assign the value to the corresponding attribute in the dataclass
            setattr(config_class, key, value)

    def _get_config_string(self, value):
        """Get the config string

        Args:
            value (str): value of the config

        Returns:
            str: string of the config
        """
        if value == "None":
            return None
        return value.replace("'", "") if value.startswith("'") else value.replace('"', "")

    def _get_config_list(self, value):
        """Get the config list
        Args:
            value (str): value of the config
        Returns:
            list: list of the config
        """
        try:
            return json.loads(value)
        except:
            return eval(value)


    def _set_new_config_path(self, path, config):
        """Set the path to the config file
            This is used during optuna tuning to set the path to the config file with the best parameters
        Args:
            path (str): path to the config file
            config (class): config class which should be set
        """
        config.path_to_config = path
