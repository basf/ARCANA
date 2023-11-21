''' This module is the main module for training the model. It contains the TrainProcedure class which is the main class'''
import os
import warnings
import json
import pickle
import numpy as np
import torch

from arcana.logger import logger
from arcana.training import train_model
from arcana.losses.loss import LossFactory
from arcana.regularizations.optimizer_scheduler import SchedulerFactory
from arcana.models.sequence_to_sequence.seq2seq_factory import Seq2SeqFactory
from arcana.procedures.config_handler import ConfigHandler
from arcana.processing.data_processing import DataPreparation
from arcana.utils import utils
# from arcana.plots import plots
warnings.filterwarnings("ignore")
# plots.Plots()
np.random.seed(0)
log = logger.get_logger("arcana.run_procedure")

SEED = 0
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class TrainProcedure:
    """This class is the main class for training the model. It contains some of the necessary functions for training,
    predicting and finetuning the model. The class also contains all the parameters for the training, predicting and
    tuning of the model. It also contains the functions for saving the model parameters and the
    data splits."""
    def __init__(self):
        config_handler = ConfigHandler()
        self.general_config = config_handler.get_general_config()
        self.data_config = config_handler.get_data_config()
        self.procedure_config = config_handler.get_procedure_config()
        self.model_config = config_handler.get_model_config()
        self.model_config.dim_weights = torch.tensor(self.model_config.dim_weights)
        # initializing the data preparation class
        self.data_preparation = DataPreparation(self.general_config, self.data_config, self.procedure_config)
        # initializing the model class
        self.device = None
        self.set_device()
        # initializing the loss class
        self.criterion = None
        # initializing the optimizer class
        self.optimizer = None
        self.scheduler = None
        # initializing the model class
        self.seq2seq_factory = Seq2SeqFactory(self.model_config)
        self.seq_2_seq_trainer = None
        #self.model = None
        self.train_parameters = None
        # initializing the loaders
        self.train_loader = None
        self.val_loader = None
        # get the data splits
        if self.general_config.pretrained_model:
            if self.procedure_config.transfer_learning:
                self.data_splits()
            if self.procedure_config.predicting and (not self.procedure_config.transfer_learning):
                pass
        else:
            self.data_splits()

        # if ((not self.procedure_config.naive_training) and (not self.procedure_config.transfer_learning) and \
        #     (not self.procedure_config.optuna_tuning) and (self.procedure_config.predicting)):
        #     self.data_splits()

    def set_device(self):
        """Set the device for training the model
        """
        # move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            torch.set_default_tensor_type("torch.FloatTensor")
        if not self.device.type == "cpu":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        log.info(f"device: {self.device}")


    def data_splits(self):
        """Get the data splits for training, validation and testing
        """
        self.data_preparation.get_data_for_model()
        self.data_preparation.prepare_data_for_model()
        self.save_data_splits()


    def save_data_splits(self):
        """Save the data splits
        """
        data_path = os. path.join(self.model_config.result_path, "data_splits")
        # save the original data splits
        for original_data, data_name in zip([self.data_preparation.df_train_original, self.data_preparation.df_val_original,
                              self.data_preparation.df_test_original], ["train", "val", "test"]):
            original_data.to_csv(os.path.join(data_path, f"{data_name}_original.csv"))

        if self.procedure_config.preprocess_data:
            # save model data transformation
            with open(os.path.join(data_path, "model_data_transformation.pkl"), "wb") as f:
                pickle.dump(self.data_preparation.model_data_transformation, f)
            # save the test_names of the test data
            np.save(os.path.join(data_path, "test_names.npy"), self.data_preparation.test_data_names)
            # save the processed data splits
            for processed_data, processed_name in zip([self.data_preparation.padded_train_data,
                                self.data_preparation.padded_val_data, self.data_preparation.padded_test_data],
                                ["train", "val", "test"]):
                torch.save(processed_data, os.path.join(data_path, f"{processed_name}_processed.pt"))

    def loader_initialization(self):
        """Initialize the data loaders
        """
        # define the data loaders
        self.train_loader = torch.utils.data.DataLoader(self.data_preparation.padded_train_data,
                                                batch_size=self.model_config.batch_size)
        self.val_loader = torch.utils.data.DataLoader(self.data_preparation.padded_val_data,
                                                batch_size=self.model_config.batch_size)


    def model_parameter_initialization(self):
        """Initialize the model parameters
        """

        # define the data loaders
        # self.train_loader = torch.utils.data.DataLoader(self.data_preparation.padded_train_data,
        #                                         batch_size=self.model_config.batch_size)
        # self.val_loader = torch.utils.data.DataLoader(self.data_preparation.padded_val_data,
        #                                         batch_size=self.model_config.batch_size)
        # define the model
        if self.procedure_config.attention_type == "additive":
            self.seq2seq_factory.create_additive_model()

        elif self.procedure_config.attention_type == "multihead":
            self.seq2seq_factory.create_multihead_model()

        # parallelize the model if more than one GPU is available
        if torch.cuda.device_count() > 1:
            log.info(f"Using {torch.cuda.device_count()} GPUs")
            self.seq2seq_factory.seq2seq = torch.nn.DataParallel(self.seq2seq_factory.seq2seq)


    def train_element_initialization(self):
        """Initialize the training elements
        """
        # define the loss
        self.criterion = LossFactory.create_loss(self.model_config)

        # define optimizer
        optimizer = torch.optim.Adam(self.seq2seq_factory.seq2seq.parameters(),
                        lr=self.model_config.learning_rate,
                        weight_decay=self.model_config.weight_decay)

        # Instantiate the factory with the optimizer and params
        scheduler_factory = SchedulerFactory(optimizer, self.model_config, len_train_loader=len(self.train_loader))
        # Get the desired scheduler
        scheduler = scheduler_factory.get_scheduler(learning_rate_type = self.procedure_config.learning_rate_type)

        # define the trainer
        self.seq_2_seq_trainer =  train_model.Seq2SeqTrainer(self.seq2seq_factory.seq2seq, self.criterion, optimizer, self.device,
                                                        scheduler, self.model_config)


    def save_model_parameters(self, model_folder = None):
        """Save the model parameters

        Args:
            model_folder (str): Path to the model folder
        """
        model_folder = model_folder if model_folder else self.model_config.result_path
        # save the tested data
        utils.save_test_data(model=self.seq_2_seq_trainer.seq2seq,
                    model_folder=model_folder,
                    test_data=self.data_preparation.padded_test_data,
                    test_lengths=self.data_preparation.test_lengths)
        # save the trained model
        torch.save(self.seq_2_seq_trainer.seq2seq,
                   os.path.join(model_folder, "model_parameters",
                        f"model_complete_{self.seq_2_seq_trainer.seq2seq.model_name}.pth"))
        torch.save(self.seq_2_seq_trainer.seq2seq.state_dict(),
                os.path.join(model_folder, "model_parameters",
                    f"model_weights_{self.seq_2_seq_trainer.seq2seq.model_name}_{self.procedure_config.attention_type}.pt"))
        # save the model parameters
        with open(os.path.join(model_folder, "train_parameters",
                    f"parameters_{self.seq_2_seq_trainer.seq2seq.model_name}.json"), "w", encoding="utf-8") as banana:
            json.dump(self.seq_2_seq_trainer.parameters_all, banana)


    def training(self, trial = None, model_folder = None):
        """Train the model

        Args:
            trial (optuna.trial.Trial): Trial object
            model_folder (str): Path to the model folder
        """
        #initialize the data loaders
        self.loader_initialization()
        # initialize the model and training parameters
        self.model_parameter_initialization()
        # initialize the training elements
        self.train_element_initialization()
        # train the model
        self.seq_2_seq_trainer.train_model(train_loader=self.train_loader, val_loader=self.val_loader,
                                    val_lengths=self.data_preparation.val_lengths, trial=trial)
        # save the model parameters
        self.save_model_parameters(model_folder=model_folder)
