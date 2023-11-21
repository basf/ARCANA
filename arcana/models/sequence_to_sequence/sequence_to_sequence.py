""" Sequence to sequence model for time series forecasting"""
import torch
from arcana.logger import logger
from arcana.models.encoders.additive_encoder import AdditiveEncoder
from arcana.models.encoders.multihead_encoder import MultiheadEncoder

log = logger.get_logger("arcana.models.seq2seq.seq2seq")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set the default tensor type
if device.type == "cpu":
    torch.set_default_tensor_type(torch.FloatTensor)
if not device.type == "cpu":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Seq2Seq(torch.nn.Module):
    """Seq2Seq module """
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.model_name = "seq2seq"
        # encoder and decoder are instances of the Encoder and Decoder classes
        self.encoder = encoder
        self.decoder = decoder
        self.window_length = config.window_length
        self.warmup_steps = config.warmup_steps
        self.train_path = config.result_path
        self.dim_depth = None
        self.attention_probs_encoder = None
        self.attention_decoder = None

    def __repr__(self):
        return f"window_length={self.window_length}, warmup_steps={self.warmup_steps},\
                train_path={self.train_path}), decoder={self.decoder}, Seq2Seq(encoder={self.encoder}"

    def forward(self, source, target, source_lengths, teacher_forcing_ratio, start_position):
        """Forward pass for seq2seq model. The forward pass is implemented as follows:
        1. get the encoder outputs
        2. iterate over the target sequence by specific window length
        3. get the prediction from the decoder
        4. concatenate the exogenous variables with the prediction
        5. store the prediction in a tensor

        Args:
            source (torch.Tensor): source tensor (batch_size, seq_length, input_size)
            target (torch.Tensor): target tensor (batch_size, seq_length, output_size)
            source_lengths (torch.Tensor): source lengths (batch_size)
            teacher_forcing_ratio (float): teacher forcing ratio
            start_position (int): start position of the prediction

        Returns:
            outputs (torch.Tensor): outputs (num_quantiles, batch_size, seq_length, output_size)
        """

        # length of the target sequence
        target_len = target.shape[1]
        # tuple to store decoder outputs
        outputs_1, outputs_2, outputs_3 = [], [], []

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        if isinstance(self.encoder, AdditiveEncoder):
            encoder_outputs, hidden, cell = self.encoder(source, source_lengths)
        elif isinstance(self.encoder, MultiheadEncoder):
            encoder_outputs, hidden, cell, self.attention_probs_encoder = self.encoder(source, source_lengths)
        else:
            raise ValueError(f"Encoder {self.encoder} is not supported")

        # start position for the decoder
        start_position = start_position - self.window_length

        x_tensor = source[:, -self.window_length :, :]

        # NOTE: this is necessary to deal with the number of exogenous variables that are used
        if self.decoder.input_size == self.decoder.output_size +1:
            self.dim_depth = 1
        elif self.decoder.input_size == self.decoder.output_size +2:
            self.dim_depth = 2
        else:
            self.dim_depth = 3

        for t_len in range(target_len):
            # get the prediction from the decoder
            prediction, hidden, cell, self.attention_decoder = self.decoder(x_tensor, hidden, cell,
                                                            encoder_outputs)
            prediction_1, prediction_2, prediction_3 = prediction

            # concatenate the time step
            prediction_1 = torch.cat((target[:,t_len,0:self.dim_depth].unsqueeze(1), prediction_1), dim=2)
            prediction_2 = torch.cat((target[:,t_len,0:self.dim_depth].unsqueeze(1), prediction_2), dim=2)
            prediction_3 = torch.cat((target[:,t_len,0:self.dim_depth].unsqueeze(1), prediction_3), dim=2)

            # store the prediction in a tensor
            outputs_1.append(prediction_1.squeeze(1).clone())
            outputs_2.append(prediction_2.squeeze(1).clone())
            outputs_3.append(prediction_3.squeeze(1).clone())

            # add warmup steps here
            if t_len < self.warmup_steps:
                if t_len + 1 < self.window_length:
                    x_tensor = torch.cat(
                        (
                            source[:, -self.window_length + t_len + 1 :, :],
                            target[:, : t_len + 1]), dim=1)
                else:
                    x_tensor = target[:, t_len + 1 - self.window_length : t_len + 1]
            else:
                # decide if we are going to use teacher forcing or not
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                # get the highest predicted token from our predictions
                if teacher_force:
                    if t_len + 1 < self.window_length:
                        x_tensor = torch.cat(
                            (
                                source[:, -self.window_length + t_len + 1 :, :],
                                target[:, : t_len + 1]), dim=1)
                    else:
                        # x = torch.cat((x, target[:, t:t+1, :]), dim=1)
                        # x = x[:, -self.window_length:, :]
                        x_tensor = target[:, t_len + 1 - self.window_length : t_len + 1]
                else:
                    x_tensor = torch.cat((x_tensor, prediction_2), dim=1)
                    x_tensor = x_tensor[:, -self.window_length :, :]

                # TODO: plot according to the type of attention (look at old code)

        outputs_1 = torch.stack(outputs_1, dim=1)
        outputs_2 = torch.stack(outputs_2, dim=1)
        outputs_3 = torch.stack(outputs_3, dim=1)
        outputs = torch.stack([outputs_1, outputs_2, outputs_3], dim=0)
        return outputs
