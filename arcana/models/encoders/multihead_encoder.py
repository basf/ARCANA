"""Multihead encoder module for the ARCANA project."""
import torch
from arcana.logger import logger
from arcana.models.encoders.base_encoder import BaseEncoder

log = logger.get_logger("arcana.models.encoders.multihead_encoder")


class MultiheadEncoder(BaseEncoder):
    """Multihead encoder module """
    def __init__(self, config):
        """
        Multihead encoder class initialization

        Args:
            config (ModelConfig): configuration class

        Raises:
            ValueError: nhead cannot be None for MultiheadEncoder, it should be an integer.
        """
        # check if nhead is None then throw error
        if config.nhead_encoder is None:
            raise ValueError("nhead cannot be None for MultiheadEncoder")

        super().__init__(config=config)
        self.nhead = config.nhead_encoder
        # add multihead attention layer
        self.multihead_attention = torch.nn.MultiheadAttention(self.hidden_size * (2 if self.bidirectional else 1),
                                                      self.nhead, batch_first=True)

    def __repr__(self):
        return f"EncoderParameter(input_size={self.input_size}, hidden_size={self.hidden_size} \
                \n dropout={self.dropout}, batch_size={self.batch_size} \
                \n num_layers={self.num_layers}, bidirectional={self.bidirectional},\
                \n num_head={self.nhead}, device={self.device}) \n" + super().__repr__()


    def forward(self, x_tensor, lengths):
        x_tensor = x_tensor.type(torch.float32) #x_tensor: (batch_size, seq_length, input_size)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_tensor, lengths.to('cpu'), batch_first=True, enforce_sorted=False).to(self.device)
        num_directions = 2 if self.bidirectional else 1
        outputs, (self.hidden, self.cell) = self.lstm(packed)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        _, attention_probs_encoder = self.multihead_attention(outputs, outputs,
                                                        outputs, average_attn_weights=False)
        # add layer normalization and skip connection to outputs and attention outputs
        #outputs = outputs_before_attention + outputs_attention
        #outputs = self.layer_norm(outputs)

        if self.bidirectional:
            self.hidden = self.hidden.view(self.num_layers, num_directions, -1, self.hidden_size)
            self.cell = self.cell.view(self.num_layers, num_directions, -1, self.hidden_size)
            self.hidden = torch.cat((self.hidden[:, 0, :, :], self.hidden[:, 1, :, :]), dim=2)
            self.cell = torch.cat((self.cell[:, 0, :, :], self.cell[:, 1, :, :]), dim=2)


        return outputs, self.hidden, self.cell, attention_probs_encoder
