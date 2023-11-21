''' Class to trace the loss'''
class LossTrace:
    """Class to trace the loss of the model during training and validation"""
    def __init__(self, output_dim) -> None:
        self.output_dim = output_dim
        self.losses = {"train_loss_batch": [], "train_loss_epoch": [],
                       "val_loss_batch": [], "val_loss_epoch": []}
        self.loss_epoch_individual_train = {f'train_loss_epoch_dim_{i}': []
                                            for i in range(self.output_dim)}
        self.loss_epoch_individual_val = { f'val_loss_epoch_dim_{i}': []
                                            for i in range(self.output_dim)}
        self.losses.update(**self.loss_epoch_individual_train, **self.loss_epoch_individual_val)

        self.temp_losses = None
        self.temp_losses_individual_train = None
        self.temp_losses_individual_val = None

    def reset_temp_loss_trace(self):
        """Reset the temporary loss trace"""
        self.temp_losses = {"temp_train_loss_epoch": 0, "temp_val_loss_epoch": 0}
        self.temp_losses_individual_train = {f'temp_train_loss_epoch_dim_{i}': 0
                                        for i in range(self.output_dim)}
        self.temp_losses_individual_val = {f'temp_val_loss_epoch_dim_{i}': 0
                                           for i in range(self.output_dim)}

    def calculate_batch_loss_train(self, train_loss):
        """Calculate the loss for each batch and add it to the loss_trace

        Args:
            train_loss (torch.Tensor): loss for each batch
        """

         # calculate the loss for each batch
        self.losses["train_loss_batch"].append(train_loss.item())
        # add the losses for later averaging for each epoch
        self.temp_losses["temp_train_loss_epoch"] += train_loss.item()

    def calculate_batch_loss_validation(self, validation_loss):
        """Calculate the validation loss for each batch and add it to the loss_trace

        Args:
            validation_loss (torch.Tensor): validation loss for each batch
        """
         # add the losses for later averaging for each epoch
        self.losses["val_loss_batch"].append(validation_loss.item())
        self.temp_losses["temp_val_loss_epoch"] += validation_loss.item()

    def calculate_epoch_loss(self, train_loader, val_loader):
        """Calculate the average loss for each epoch and add it to the loss_trace

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
            val_loader (torch.utils.data.DataLoader): validation data loader
        """
        # calculate the average loss for each epoch
        self.losses["train_loss_epoch"].append(
            self.temp_losses["temp_train_loss_epoch"]/len(train_loader))
        self.losses["val_loss_epoch"].append(
            self.temp_losses["temp_val_loss_epoch"]/len(val_loader))

        # add the individual scores and losses of both training and validation for each epoch
        for i in range(self.output_dim):
            # # losses
            self.losses[f'train_loss_epoch_dim_{i}'].append(
                self.temp_losses_individual_train[f'temp_train_loss_epoch_dim_{i}']/len(train_loader))
            self.losses[f'val_loss_epoch_dim_{i}'].append(
                self.temp_losses_individual_val[f'temp_val_loss_epoch_dim_{i}']/len(val_loader))
