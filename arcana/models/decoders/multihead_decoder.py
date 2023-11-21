"""Multihead dencoder module for the ARCANA project. """
import torch
from arcana.logger import logger
from arcana.models.decoders.base_decoder import BaseDecoder, initialize_weights

log = logger.get_logger("arcana.models.decoders.multihead_decoder")

class MultiheadDecoder(BaseDecoder):
    """Multihead decoder module """
    def __init__(self, config):
        """
        Multihead decoder class initialization

        Args:
            config (ModelConfig): configuration

        Raises:
            ValueError: nhead cannot be None for MultiheadDecoder, it should be an integer.
        """
        # check if nhead is None then throw error
        if config.nhead_decoder is None:
            raise ValueError("nhead cannot be None for MultiheadDecoder")

        super().__init__(config)
        self.nhead = config.nhead_decoder
        # add multihead attention layer
        self.multihead_attention = torch.nn.MultiheadAttention(self.hidden_size * (2 if self.bidirectional_encoder else 1),
                                                      self.nhead, batch_first=True)
        # initialize weights
        self.apply(initialize_weights)

    def __repr__(self):
        return f"DecoderParameter(input_size={self.input_size}, hidden_size={self.hidden_size} \
                \n output_size={self.output_size}, lstm_dropout={self.lstm_dropout} \
                \n num_layers={self.num_layers}, bidirectional_encoder={self.bidirectional_encoder}, \
                \n nhead={self.nhead}, output_dropout={self.output_dropout}, device={self.device}) \n" + super().__repr__()

    def forward(self, x_tensor, hidden_state, cell_state, encoder_outputs):
        """Forward pass for multihead decoder module.

        The forward pass is implemented as follows:
            1. get attention scores
            2. concatenate the attention scores with the input tensor
            3. pass the concatenated tensor through the lstm layer
            4. pass the lstm output through the fc layer
            5. pass the fc layer output through the leaky relu layer
            6. pass the leaky relu output through the output dropout layer
            7. pass the output dropout layer output through the fc layer for quantile 1, 2 and 3
            8. concatenate the quantile 1, 2 and 3 predictions

        Args:
            x_tensor (torch.Tensor): input tensor (batch_size, seq_length, input_size)
            hidden_state (torch.Tensor): hidden state (num_layers, batch_size, hidden_size)
            cell_state (torch.Tensor): cell state (num_layers, batch_size, hidden_size)
            encoder_outputs (torch.Tensor): encoder outputs (batch_size, seq_length, hidden_size)

        Returns:
            predictions (tuple): tuple of quantile predictions
            hidden_out (torch.Tensor): hidden state
            cell_out (torch.Tensor): cell state
            attention_output_weights_probs_decoder (torch.Tensor): attention scores

        """
        attention_outputs, attention_output_weights_probs_decoder =\
                self.multihead_attention(hidden_state.sum(dim=0).unsqueeze(1),encoder_outputs,
                                encoder_outputs, average_attn_weights=False)
        x_tensor = x_tensor.type(torch.float32)
        #x = self.dropout(x)
        lstm_input = torch.cat((x_tensor, attention_outputs.repeat(1, x_tensor.size(1), 1)), dim=2)
        #residual = self.residual_fc(x)

        if self.num_layers == hidden_state.shape[0]:
            hidden_init, cell_init = hidden_state, cell_state
        else:
            hidden_init = hidden_state.sum(dim=0).unsqueeze(0).repeat(self.num_layers, 1, 1)
            cell_init = cell_state.sum(dim=0).unsqueeze(0).repeat(self.num_layers, 1, 1)

        outputs, (hidden_out, cell_out) = self.lstm(lstm_input, (hidden_init, cell_init))
        # apply skip connection
        #outputs = outputs + residual

        outputs = self.fc_layer(outputs) # (batch_size, seq_length, hidden_size)
        outputs = self.leaky_relu(outputs)
        outputs = self.output_dropout(outputs)

        # quantile predictions
        predictions_quantile_1 = self.fc_layer_pred_1(outputs[:, -1, :].unsqueeze(1))
        predictions_quantile_2 = self.fc_layer_pred_2(outputs[:, -1, :].unsqueeze(1))
        predictions_quantile_3 = self.fc_layer_pred_3(outputs[:, -1, :].unsqueeze(1))
        predictions = (predictions_quantile_1, predictions_quantile_2, predictions_quantile_3)

        return predictions, hidden_out, cell_out, attention_output_weights_probs_decoder
