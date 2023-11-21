"""Base encoder class"""
import torch
from arcana.logger import logger

log = logger.get_logger("arcana.models.encoders.base_encoder")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set the default tensor
if device.type == "cpu":
    torch.set_default_tensor_type(torch.FloatTensor)
if not device.type == "cpu":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class BaseEncoder(torch.nn.Module):
    """Base encoder module """
    def __init__(self, config):
        """Base encoder class initialization

        Args:
            config (ModelConfig): configuration class

        """
        super().__init__()
        self.device = device
        self.input_size = config.input_size
        self.hidden_size = config.hidden_dim
        # add dropout layer
        self.dropout = torch.nn.Dropout(config.dropout_encoder)
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers_encoder
        self.bidirectional = config.bidirectional
        # add lstm layer
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                    dropout=config.dropout_encoder, bidirectional=self.bidirectional)
        # add layer normalization
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size * (2 if self.bidirectional else 1))
        self.cell = None
        self.hidden = None


    def reset_hidden_state(self):
        """ Reset the hidden state of the LSTM """
        self.hidden = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                                  self.batch_size, self.hidden_size).to(device)
        self.cell = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                                self.batch_size, self.hidden_size).to(device)


    def forward(self, x_tensor, lengths):
        """Forward pass to be implemented by subclass"""
        raise NotImplementedError("This method should be overridden by subclass")
