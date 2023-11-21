'''This script contains the training function for the sequence to sequence model'''
import os
import json
import torch
import optuna

from tqdm import tqdm

from arcana.plots import plot_utils
from arcana.regularizations import stopping_criteria, teacher_forcing
from arcana.losses.loss_trace import LossTrace
from arcana.losses.loss import PinballLoss, CombinedHPLoss, CombinedLPLoss
from arcana.plots import plots
from arcana.logger import logger


plots.Plots()
log = logger.get_logger("arcana.training.train_model")


class Seq2SeqTrainer:
    """Class for training the sequence to sequence model"""
    def __init__(self, seq2seq, criterion, optimizer, device, scheduler, config):
        """Initialize the Seq2SeqTrainer class needed for training the model

        Args:
            seq2seq (torch.nn.Module): model
            criterion (torch.nn.Module): loss function
            optimizer (torch.optim): optimizer
            device (torch.device): device, cpu or gpu
            scheduler (torch.optim.lr_scheduler): scheduler
            config (dict): Configuration dictionary containing all required parameters
        """
        self.seq2seq = seq2seq      # model
        self.criterion = criterion  # loss function
        self.loss_type = type(criterion).__name__
        self.scheduler = scheduler  # scheduler
        self.learning_rate_type = type(scheduler).__name__
        self.optimizer = optimizer
        self.device = device        # device, cpu or gpu
        self.config = config  # Configuration dictionary containing all required parameters
        self.loss_trace = LossTrace(self.config.output_size)
        self.early_stopping = None
        self.teacher_forcing_ratio_list = None
        self.available_sequence_list = None
        self.learning_rate_dict = {'learning_rate': [], 'epoch': []}
        self.parameters_all = None
        self.temp_path = None
        self.parameters_path = None


    def initialize_training(self):
        """Initialize different parameters for training the model:
        - create directories for saving the model and parameters
        - move the model to the device
        - initialize early stopping and teacher forcing
        """
        self.temp_path = os.path.join(self.config.result_path, "temp_models")
        self.parameters_path = os.path.join(self.config.result_path, "train_parameters")

        # Move the model to the device
        self.seq2seq.to(self.device)

        # Initialize early stopping
        self.early_stopping = stopping_criteria.EarlyStopping(
            criterion_rule=self.config.early_stopping_type,
            training_strip=5,  # This can be parameterized
            alpha=self.config.early_stopping_alpha,
            patience=self.config.patience)

        # Initialize teacher forcing
        self.teacher_forcing_ratio_list, self.available_sequence_list =\
        teacher_forcing.TeacherForcingScheduler(num_epochs = self.config.number_of_epochs+1,
                                                epoch_division = self.config.epoch_division,
                                                seq_start = self.config.minimum_cycle_length,
                                                seq_end = self.config.maximum_cycle_length,
                                                start_learning_ratio = self.config.tl_start_ratio,
                                                end_learning_ratio = self.config.tl_end_ratio,
                                                decay_stride = self.config.decay_stride).get_ratio()


    def early_stopping_check(self, train_loss, val_loss, epoch):
        """Check if early stopping should be applied

        Args:
            train_loss (float): training loss
            val_loss (float): validation loss
            epoch (int): current epoch
        Returns:
            should_stop (bool): True if early stopping should be applied
        """
        should_stop = self.early_stopping.step(train_loss, val_loss, epoch)
        return should_stop


    def train_epoch(self, epoch, train_loader, available_seq):
        """Train the model for one epoch by looping through the batches
        - zero the hidden states and gradients
        - forward pass
        - compute loss and backpropagation
        - update parameters and learning rate
        - calculate the loss for each batch and add it to the loss_trace

        Args:
            epoch (int): current epoch
            train_loader (torch.utils.data.DataLoader): training data loader
            available_seq (int): available sequence length
        """
        self.seq2seq.train()
        # Logic for training one epoch
        for _, (data_train) in enumerate(train_loader):
            # get the inputs and labels
            source = data_train[:, :available_seq, :]
            # add the device to the source
            source.to(self.device)
            # get the target
            #FIXME: this length should be dynamic
            target = data_train[:, available_seq:, :] # self.config['last_step'] general: maximum available cycle
            target = torch.tensor(target).type(torch.float32)
            target.to(self.device)
            # get the lengths of the source
            source_lengths = [len(s) for s in source]
            source_lengths = torch.tensor(source_lengths).type(torch.int32)

            # Zero the hidden states
            self.seq2seq.encoder.reset_hidden_state()

            # Zero the parameter gradients to avoid accumulation
            self.optimizer.zero_grad()

            # Forward pass
            output = self.seq2seq(source = source, target = target, source_lengths = source_lengths,
                                teacher_forcing_ratio=self.teacher_forcing_ratio_list[epoch],
                                start_position=available_seq)

            # Compute loss
            loss = self.compute_loss(output = output, target = target, process_type="train")

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.seq2seq.parameters(), max_norm=1)

            # Update parameters
            self.optimizer.step()

            # Update the learning rate
            if self.learning_rate_type != "ReduceLROnPlateau":
                self.scheduler.step()

            self.loss_trace.calculate_batch_loss_train(loss)


    def validation_epoch(self, val_loader, val_lengths, available_seq):
        """Validate the model for one epoch by looping through the batches
        - turn off the gradients
        - zero the hidden states
        - forward pass and compute loss
        - calculate the loss for each batch and add it to the loss_trace

        Args:
            val_loader (torch.utils.data.DataLoader): validation data loader
            val_lengths (list): list of lengths of the validation data
            available_seq (int): available sequence length
        """
        # validate the model
        self.seq2seq.eval()
        with torch.no_grad():
            for batch_idx_val, (data_val) in enumerate(val_loader):
                # get the inputs and labels
                val_source = data_val[:, :available_seq, :]
                val_source.to(self.device)
                #FIXME: this length should be dynamic
                val_target = data_val[:, available_seq:, :] # self.config['last_step'] general: maximum available cycle
                val_target = torch.tensor(val_target).type(torch.float32)
                val_target.to(self.device)
                try:
                    val_length_loader = val_lengths[batch_idx_val *
                                        self.config.batch_size:(batch_idx_val+1)*self.config.batch_size]
                except:
                    val_length_loader = val_lengths[batch_idx_val*self.config.batch_size:]

                val_source_lengths =\
                    torch.tensor([min(val_lengths[batch_idx_val*self.config.batch_size:(batch_idx_val+1)*self.config.batch_size][i],
                                  torch.tensor(available_seq)) for i in range(len(val_length_loader))])

                # Zero the hidden states
                self.seq2seq.encoder.reset_hidden_state()

                # Forward pass
                output_val = self.seq2seq(val_source, val_target, val_source_lengths,
                                     teacher_forcing_ratio = 0.0, start_position = available_seq)

                # Compute loss
                validation_loss = self.compute_loss(output = output_val, target = val_target, process_type="val")


                # add the losses for later averaging for each epoch
                self.loss_trace.calculate_batch_loss_validation(validation_loss)


    def calculate_overall_loss(self, target, output, dim_weights):
        """calculating the loss function for each dimension and overall loss

        Args:
            target (tensor): target tensor
            output (tensor): predicted tensor
            dim_weights (list): weights for each dimension

        Returns:
            loss: overall loss
            temp_losses_individual: loss for each dimension
        """
        dim_weights = dim_weights.to(self.device)
        loss = 0

        for i in range(self.loss_trace.output_dim):
            start_dim = target.shape[2] - self.loss_trace.output_dim + i
            # pass mask to the loss function
            mask =  target[:, :, i].ne(0)

            individual_dim_loss = self.criterion(output[:, :, start_dim:start_dim+1],
                                            target[:, :, start_dim:start_dim+1]) * dim_weights[i]

            masked_loss = individual_dim_loss.masked_select(mask.unsqueeze(2))

            loss += masked_loss.mean()

        return loss


    def calculate_overall_combined_loss(self, target, output, process_type,
                                                        temp_losses_individual):
        """calculating the loss function for each dimension and overall loss
        - calculate the loss for endogenous variables
        - mask-out the zero values
        - calculate losses and store them in the appropriate dictionary

        Args:
            target (tensor): target tensor
            output (tensor): predicted tensor
            process_type (str): train or val
            temp_losses_individual (dict): dictionary for storing the losses for each dimension

        Returns:
            loss: overall loss
            temp_losses_individual: loss for each dimension
        """
        loss = 0
        for i in range(self.loss_trace.output_dim):
            start_dim = target.shape[2] - self.loss_trace.output_dim + i

            mask =  target[:, :, start_dim].ne(0)

            individual_loss =\
                self.criterion(output[:,:, :, start_dim:start_dim+1], target[:, :, start_dim:start_dim+1], mask)

            # store the losses
            temp_losses_individual[f'temp_{process_type}_loss_epoch_dim_{i}'] += individual_loss.item()

            loss +=  individual_loss

        return loss, temp_losses_individual


    def compute_loss(self, output, target, process_type):
        """Compute the loss for each batch
        - calculate the loss for each dimension and overall loss
        - add the losses for later averaging for each epoch

        Args:
            output (tensor): predicted tensor
            target (tensor): target tensor
            process_type (str): train or val

        Returns:
            loss: overall loss
        """
        #isinstance(self.criterion, PinballLoss) or isinstance(self.criterion, CombinedHPLoss) or isinstance(self.criterion, CombinedLPLoss)
        if isinstance(self.criterion, (CombinedHPLoss, CombinedLPLoss, PinballLoss)):
            if process_type == "train":
                loss, self.loss_trace.temp_losses_individual_train=\
                    self.calculate_overall_combined_loss(target = target, output = output, process_type= process_type,
                            temp_losses_individual= self.loss_trace.temp_losses_individual_train)
            elif process_type == "val":
                loss, self.loss_trace.temp_losses_individual_val=\
                    self.calculate_overall_combined_loss(target = target, output = output, process_type= process_type,
                            temp_losses_individual = self.loss_trace.temp_losses_individual_val)

        else:

            loss = self.calculate_overall_loss(target = target, output = output, dim_weights = self.config.dim_weights)

        return loss


    def update_training_params(self, epoch):
        """Update the learning rate and add it to the learning_rate_dict

        Args:
            epoch (int): current epoch
        """
        self.learning_rate_dict['learning_rate'].append(
                self.optimizer.param_groups[0]['lr'])
        self.learning_rate_dict['epoch'].append(epoch)


    def plot_training_params(self):
        """Plot the learning rate and losses"""
        plot_utils.plot_model_learning_rate(self.learning_rate_dict, self.parameters_path)
        plot_utils.plot_train_val_loss(
            self.loss_trace.losses, self.parameters_path, self.loss_type, train_loss_mode="batch")
        plot_utils.plot_train_val_loss(
            self.loss_trace.losses, self.parameters_path, self.loss_type, train_loss_mode="epoch")

        for i in range(self.loss_trace.output_dim):
            plot_utils.plot_train_val_loss_individual(self.loss_trace.losses, self.parameters_path,
                                    self.loss_type, train_loss_mode = f"dim_{i}")


    def save_training_results_and_plots(self, epoch = None):
        """ Save the model and parameters and plot the results

        Args:
            epoch (int): current epoch
        """

        # save the parameters (losses, scores, learning rate in a json file)
        self.parameters_all = {"losses": self.loss_trace.losses,
                            "learning_rate": self.learning_rate_dict}

        #if trial is None:

        name = f"temp_best_model_weights_{epoch+1}.pt" if epoch is not None else "temp_best_model_weights_complete.pt"

        torch.save(self.seq2seq.state_dict(), os.path.join(
                        self.temp_path, name))

        with open(os.path.join(self.parameters_path, "parameters_all.json"), "w", encoding='utf-8') as f:
            json.dump(self.parameters_all, f)
        # plot the results
        self.plot_training_params()


    def count_parameters(self):
        """Count the number of trainable parameters"""
        # get the number of traineable parameters
        return sum(p.numel() for p in self.seq2seq.parameters() if p.requires_grad)


    def train_model(self, train_loader, val_loader, val_lengths, trial=None):
        """The main function that controls the training process which does the following:
        - initialize the training
        - train the model
        - validate the model
        - calculate the loss for each epoch and add it to the loss_trace
        - print the last losses and scores after every 50 epochs
        - early stopping
        - update the training parameters
        - save the training results and plots

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
            val_loader (torch.utils.data.DataLoader): validation data loader
            val_lengths (list): list of lengths of the validation data
            trial (optuna.trial): optuna trial
        """
        log.info(f"start training with device: {self.device}")

        # initialize the training
        self.initialize_training()
        self.seq2seq.train()

        log.info(f"number of trainable parameters: {self.count_parameters()}")

        # train the model
        for epoch in tqdm(range(self.config.number_of_epochs)):
            # Reset the temp loss trace for each epoch
            self.loss_trace.reset_temp_loss_trace()
            available_seq = self.available_sequence_list[epoch]
            # train the model
            self.train_epoch(epoch, train_loader, available_seq)
            # validate the model
            self.validation_epoch(val_loader, val_lengths, available_seq)

            self.loss_trace.calculate_epoch_loss(train_loader, val_loader)

            # print the last losses and scores after every 50 epochs
            if (epoch+1) % 20 == 0:
                # Constructing the log message in multiple steps
                epoch_info = f"Epoch {epoch+1}/{self.config.number_of_epochs}"
                train_loss_info = f"train loss: {self.loss_trace.losses['train_loss_epoch'][-1]:.6f}"
                val_loss_info = f"val loss: {self.loss_trace.losses['val_loss_epoch'][-1]:.6f}"
                log_message = f"{epoch_info} - {train_loss_info} - {val_loss_info}"
                log.info(log_message)

            # early stopping
            should_stop = self.early_stopping_check(train_loss = self.loss_trace.losses['train_loss_epoch'][-1],
                            val_loss = self.loss_trace.losses['val_loss_epoch'][-1], epoch=epoch+1)


            if should_stop:
                log.info(f"Early stopping after {epoch+1} epochs and no improvements for {self.config.patience} epochs")
                self.save_training_results_and_plots(epoch = epoch)
                break

            if self.learning_rate_type == "ReduceLROnPlateau":
                self.scheduler.step(self.loss_trace.losses['val_loss_epoch'][-1])

            self.update_training_params(epoch)

            # TODO optuna part
            if trial is not None:
                # Add prune mechanism
                trial.report(self.loss_trace.losses["val_loss_epoch"][-1], epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        self.save_training_results_and_plots()
