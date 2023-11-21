"""Additive encoder module for the ARCANA project. """
import torch
from arcana.logger import logger
from arcana.models.encoders.base_encoder import BaseEncoder

log = logger.get_logger("arcana.models.encoders.additive_encoder")

class AdditiveEncoder(BaseEncoder):
    """Additive encoder module """
    def __init__(self, config):
        """Additive encoder class initialization

        Args:
            config (ModelConfig): configuration class
        """
        super().__init__(config)
        self.skip_connection = torch.nn.Linear(self.input_size, self.hidden_size * (2 if self.bidirectional else 1))


    def __repr__(self):
        return f"EncoderParameter(input_size={self.input_size}, hidden_size={self.hidden_size} \
                \n dropout={self.dropout}, batch_size={self.batch_size} \
                \n num_layers={self.num_layers}, bidirectional={self.bidirectional}, device={self.device}) \n" + super().__repr__()


    def forward(self, x_tensor, lengths):
        """Forward pass to be implemented by subclass for additive encoder

        Args:
            x_tensor (torch.Tensor): input tensor
            lengths (torch.Tensor): lengths of the input sequences
        Returns:
            outputs (torch.Tensor): output tensor
            self.hidden (torch.Tensor): hidden state
            self.cell (torch.Tensor): cell state
        """
        x_tensor = x_tensor.type(torch.float32)
        # x = self.dropout(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_tensor, lengths.to("cpu"), batch_first=True, enforce_sorted=False).to(self.device)
        num_directions = 2 if self.bidirectional else 1
        outputs, (self.hidden, self.cell) = self.lstm(packed)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # apply skip connection
        outputs = outputs + self.skip_connection(x_tensor)
        # add layer normalization to outputs
        outputs = self.layer_norm(outputs)

        if self.bidirectional:
            self.hidden = self.hidden.view(self.num_layers, num_directions, -1, self.hidden_size)
            self.cell = self.cell.view(self.num_layers, num_directions, -1, self.hidden_size)
            self.hidden = torch.cat((self.hidden[:, 0, :, :], self.hidden[:, 1, :, :]), dim=2)
            self.cell = torch.cat((self.cell[:, 0, :, :], self.cell[:, 1, :, :]), dim=2)

        return outputs, self.hidden, self.cell
