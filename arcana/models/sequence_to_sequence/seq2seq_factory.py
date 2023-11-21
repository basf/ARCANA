""" Factory class for creating Seq2Seq models. """
import torch
from arcana.logger import logger
from arcana.models.encoders import additive_encoder, multihead_encoder
from arcana.models.decoders import additive_decoder, multihead_decoder
from arcana.models.sequence_to_sequence import sequence_to_sequence

log = logger.get_logger("arcana.models.sequence_to_sequence.seq2seq_factory")

class Seq2SeqFactory:
    """Factory class for creating Seq2Seq models."""
    def __init__(self, config):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        log.info(f"device: {self.device}")

        self.config = config
        self.seq2seq = None

    def create_additive_model(self):
        """Create an additive model.

        Args:
            config (dict): Dictionary containing the configuration parameters

        Returns:
            seq2seq (Seq2Seq): The additive model
        """
        encoder = additive_encoder.AdditiveEncoder(self.config).to(self.device)
        log.info(repr(encoder))

        decoder = additive_decoder.AdditiveDecoder(self.config).to(self.device)
        log.info(repr(decoder))

        self.seq2seq = sequence_to_sequence.Seq2Seq(
            encoder=encoder, decoder=decoder, config = self.config).to(self.device)
        log.info(repr(self.seq2seq))

        # TODO is return needed?
        return self.seq2seq


    def create_multihead_model(self):
        """Create a multihead model.

        Returns:
            seq2seq (Seq2Seq): The multihead model
        """
        encoder = multihead_encoder.MultiheadEncoder(self.config).to(self.device)
        log.info(repr(encoder))

        decoder = multihead_decoder.MultiheadDecoder(self.config).to(self.device)
        log.info(repr(decoder))

        self.seq2seq = sequence_to_sequence.Seq2Seq(
            encoder=encoder, decoder=decoder, config=self.config).to(self.device)
        log.info(repr(self.seq2seq))

        return self.seq2seq


    def print_weights(self, layer):
        """Print the weights of a layer.

        Args:
            layer (torch.nn.Module): The layer to print the weights of
        """
        if isinstance(layer, torch.nn.LSTM):
            for name, param in layer.named_parameters():
                log.info(f"name: {name}, param: {param.data}")


    def count_parameters(self):
        """Count the number of trainable parameters in a model.

        Returns:
            num_params (int): The number of trainable parameters
        """
        # Get the number of trainable parameters
        return sum(p.numel() for p in self.seq2seq.parameters() if p.requires_grad)
