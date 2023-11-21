'''This module contains the class for fine-tuning the model.'''
import copy
import torch

from arcana.logger import logger
from arcana.procedures.training import TrainProcedure

log = logger.get_logger("arcana.procedures.finetuning")


class FineTuning(TrainProcedure):
    """This class is the main class for fine-tuning the model. It inherits from the TrainProcedure class.
    """
    def __init__(self, tl_strategy="decoder") -> None:
        # super class
        super().__init__()
        # load the model from the path
        self.pretrained_model = self.load_model()
        self.tl_strategy = tl_strategy
        if self.tl_strategy not in ["decoder", "fully_connected", "fc_and_attention"]:
            raise ValueError("The transfer learning strategy should be either: "
                             "'decoder', 'fully_connected', or 'fc_and_attention'")
        # prepare the data
        #self.data_splits()


    def load_model(self):
        """Load the model from the path"""
        return torch.load(self.general_config.pretrained_model)


    def unfreeze_decoder(self):
        """Freeze the encoder"""
        
        list_to_unfreeze = ["encoder.lstm.weight_ih_l2",
                            "encoder.lstm.weight_hh_l2",
                            "encoder.lstm.bias_ih_l2",
                            "encoder.lstm.bias_hh_l2"]
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for name, param in self.pretrained_model.named_parameters():
            if name in list_to_unfreeze:
                param.requires_grad = True

    def unfreeze_fully_connected(self):
        """Freeze the fully connected layer"""
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_model.decoder.fc_layer_pred_1.parameters():
            param.requires_grad = True
        for param in self.pretrained_model.decoder.fc_layer_pred_2.parameters():
            param.requires_grad = True
        for param in self.pretrained_model.decoder.fc_layer_pred_3.parameters():
            param.requires_grad = True


    def unfreeze_fc_and_attention(self):
        """Freeze the fully connected layer and the attention layer in the decoder"""
        # FIXME: fix the attention for multihead
        self.unfreeze_fully_connected()
        for name, param in self.pretrained_model.decoder.named_parameters():
            if "attention" in name:
                param.requires_grad = True

    def training(self, trial=None, model_folder = None):
        #FIXME:fix the docstring
        """Finetune the model depending on the transfer learning strategy
        Args:
            trial (optuna.trial): optuna trial
            model_folder (str): model folder
        """

        #initialize the data loaders
        self.loader_initialization()
        # freeze the layers
        if self.tl_strategy == "decoder":
            self.unfreeze_decoder()
        if self.tl_strategy == "fully_connected":
            self.unfreeze_fully_connected()
        if self.tl_strategy == "fc_and_attention":
            self.unfreeze_fc_and_attention()
        # overwrite the seq2seq model
        self.seq2seq_factory.seq2seq = copy.deepcopy(self.pretrained_model)

        # initialize the training elements
        self.train_element_initialization()
        # train the model
        self.seq_2_seq_trainer.train_model(train_loader=self.train_loader, val_loader=self.val_loader,
                                    val_lengths=self.data_preparation.val_lengths, trial=trial)
        # overwrite the a
        # save the model parameters
        self.save_model_parameters(model_folder=model_folder)
