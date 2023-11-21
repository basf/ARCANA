"""Additive dencoder module for the ARCANA project. """
import torch
import torch.nn.functional as F
from arcana.logger import logger
from arcana.models.decoders.base_decoder import BaseDecoder, initialize_weights

log = logger.get_logger("arcana.models.decoders.additive_decoder")


class AdditiveDecoder(BaseDecoder):
    """Additive decoder module """
    def __init__(self, config):
        """Additive decoder class initialization

        Args:
            config (ModelConfig): configuration class
        """
        super().__init__(config)

        self.attention = Attention(self.hidden_size * (2 if self.bidirectional_encoder else 1))
        # initialize weights
        self.apply(initialize_weights)


    def __repr__(self):
        return f"DecoderParameter(input_size={self.input_size}, hidden_size={self.hidden_size} \
                \n output_size={self.output_size}, lstm_dropout={self.lstm_dropout} \
                \n num_layers={self.num_layers}, bidirectional_encoder={self.bidirectional_encoder}, \
                \n output_dropout={self.output_dropout}, device={self.device}) \n" + super().__repr__()


    def forward(self, x_tensor, hidden_state, cell_state, encoder_outputs):
        """
        Forward pass for additive decoder module. The forward pass is implemented as follows:

            1. get attention scores
            2. create context vector
            3. concatenate the context vector with the input tensor
            4. pass the concatenated tensor through the lstm layer
            5. pass the lstm output through the fc layer
            6. pass the fc layer output through the leaky relu layer
            7. pass the leaky relu output through the output dropout layer
            8. pass the output dropout layer output through the fc layer for quantile 1, 2 and 3
            9. concatenate the quantile 1, 2 and 3 predictions

        Args:
            x_tensor (torch.Tensor): input tensor
            hidden_state (torch.Tensor): hidden state
            cell_state (torch.Tensor): cell state
            encoder_outputs (torch.Tensor): encoder outputs
        Returns:
            predictions (tuple): tuple of quantile predictions
            hidden_out (torch.Tensor): hidden state
            cell_out (torch.Tensor): cell state
            attention_scores (torch.Tensor): attention scores
        """

        attention_scores = self.attention(hidden_state.sum(dim=0), encoder_outputs)
        # break if attention scores are nan
        if torch.isnan(attention_scores).any():
            log.error("Attention scores are nan. Try higher learning rate. Exiting...")
            raise ValueError("Attention scores are nan. Try higher learning rate. Exiting...")
        # create context vector
        context_vector = torch.bmm(attention_scores.unsqueeze(1), encoder_outputs)
        # create a tensor of size (batch_size, 1, hidden_size)
        x_tensor = x_tensor.type(torch.float32)
        lstm_input = torch.cat((x_tensor, context_vector.repeat(1, x_tensor.size(1), 1)), dim=2)

        # initialize hidden and cell state according to the number of layers of the encoder
        if self.num_layers == hidden_state.shape[0]:
            hidden_init, cell_init = hidden_state, cell_state
        else:
            hidden_init = hidden_state.sum(dim=0).unsqueeze(0).repeat(self.num_layers, 1, 1)
            cell_init = cell_state.sum(dim=0).unsqueeze(0).repeat(self.num_layers, 1, 1)

        outputs, (hidden_out, cell_out) = self.lstm(lstm_input, (hidden_init, cell_init))

        outputs = self.fc_layer(outputs)
        outputs = self.leaky_relu(outputs)
        outputs = self.output_dropout(outputs)
        # get quantile predictions
        predictions_quantile_1 = self.fc_layer_pred_1(outputs[:, -1, :].unsqueeze(1))
        predictions_quantile_2 = self.fc_layer_pred_2(outputs[:, -1, :].unsqueeze(1))
        predictions_quantile_3 = self.fc_layer_pred_3(outputs[:, -1, :].unsqueeze(1))
        # concatenate the predictions
        predictions = (predictions_quantile_1, predictions_quantile_2, predictions_quantile_3)

        return predictions, hidden_out, cell_out, attention_scores


# create attention class
class Attention(torch.nn.Module):
    """Additive attention module """
    def __init__(self, hidden_size):
        super().__init__()
        # W1 is used to transform the hidden state of the decoder
        self.W1 = torch.nn.Linear(hidden_size, hidden_size)
        # W2 is used to transform the hidden state of the encoder
        self.W2 = torch.nn.Linear(hidden_size, hidden_size)
        # V is used to transform the tanh of W1*ht-1 + W2*hs (whre ht-1 is the hidden state of
        # the decoder at the previous time step and hs is the hidden state of the encoder at the
        # current time step)
        self.V = torch.nn.Linear(hidden_size, 1)

        #self.apply(initialize_weights)

    def forward(self, hidden, encoder_outputs):
        """Forward pass for additive attention module

        The forward pass is implemented as follows:

            1. transform the hidden state of the decoder to the same size as the hidden state of the encoder
            2. add the transformed hidden state of the decoder and the hidden state of the encoder
            3. apply tanh activation to the sum
            4. transform the tanh output to a scalar
            5. apply softmax to the scalar

        Args:
            hidden (torch.Tensor): hidden state of the decoder. The shape is (batch_size, hidden_size)
            encoder_outputs (torch.Tensor): hidden state of the encoder. The shape is (batch_size, seq_length, hidden_size)
        Returns:
            attention_scores (torch.Tensor): attention scores. The shape is (batch_size, seq_length)
        """
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        energy = torch.tanh(
            self.W1(hidden) + self.W2(encoder_outputs)
        )  # (batch_size, seq_length, hidden_size)
        attention_scores = self.V(energy).squeeze(2)  # (batch_size, seq_length)
        return F.softmax(attention_scores, dim=1)
