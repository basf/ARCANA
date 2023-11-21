"""Base decoder class"""
import torch
from arcana.logger import logger

log = logger.get_logger("arcana.models.encoders.base_decoder")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set the default tensor
if device.type == "cpu":
    torch.set_default_tensor_type(torch.FloatTensor)
if not device.type == "cpu":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class BaseDecoder(torch.nn.Module):
    """Base decoder module """
    def __init__(self, config):
        """Base decoder class initialization

        Args:
            config (ModelConfig): configuration class
        """
        super().__init__()
        self.device = device
        self.input_size = config.input_size
        self.hidden_size = config.hidden_dim
        self.output_size = config.output_size
        self.lstm_dropout = config.dropout_decoder
        # add dropout layer
        self.output_dropout = torch.nn.Dropout(config.output_dropout)
        self.num_layers = config.num_layers_decoder
        self.bidirectional_encoder = config.bidirectional
        # add lstm layer
        self.lstm = torch.nn.LSTM(
            self.input_size + self.hidden_size * (2 if self.bidirectional_encoder else 1),
            self.hidden_size * (2 if self.bidirectional_encoder else 1),
            self.num_layers, batch_first=True, dropout=self.lstm_dropout)
        # add layer normalization
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size * 2 * (2 if self.bidirectional_encoder else 1))
        # fc_layer layer
        self.fc_layer = torch.nn.Linear(
            self.hidden_size * (2 if self.bidirectional_encoder else 1),
            self.hidden_size * (2 if self.bidirectional_encoder else 1))
        # leaky relu layer
        self.leaky_relu = torch.nn.LeakyReLU()
        # fc_layer_pred_1 layer
        self.fc_layer_pred_1 = torch.nn.Linear(
            self.hidden_size * (2 if self.bidirectional_encoder else 1), self.output_size)
        # fc_layer_pred_2 layer
        self.fc_layer_pred_2 = torch.nn.Linear(
            self.hidden_size * (2 if self.bidirectional_encoder else 1), self.output_size)
        # fc_layer_pred_3 layer
        self.fc_layer_pred_3 = torch.nn.Linear(
            self.hidden_size * (2 if self.bidirectional_encoder else 1), self.output_size)


    def forward(self, x_tensor, hidden_state, cell_state, encoder_outputs):
        """Forward pass to be implemented by subclass"""
        raise NotImplementedError("This method should be overridden by subclass")


def initialize_weights(layer):
    """Initialize weights for the layer"""
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)

        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
