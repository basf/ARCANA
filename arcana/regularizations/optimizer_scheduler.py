''' Optimizer and Scheduler Factory '''
from torch import optim

class SchedulerFactory:
    """Factory class for the scheduler"""
    def __init__(self, optimizer, model_config, len_train_loader = None):
        self.optimizer = optimizer
        self.model_config = model_config
        self.len_train_loader = len_train_loader

    def get_scheduler(self, learning_rate_type):
        """Get the scheduler

        Args:
            learning_rate_type (str): learning rate type

        Returns:
            torch.optim: scheduler of the given type
        Raises:
            ValueError: if the learning rate type is unknown
        """

        if learning_rate_type == "reduced":
            return self._reduced_lr_scheduler()
        if learning_rate_type == "cycle":
            return self._cyclic_lr_scheduler()
        if learning_rate_type == "onecycle":
            return self._one_cycle_lr_scheduler()
        raise ValueError(f"Unknown learning rate type: {learning_rate_type}")

    def _reduced_lr_scheduler(self):
        """Get the reduced learning rate scheduler

        Returns:
            torch.optim: reduced learning rate scheduler
        """
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.model_config.factor_reduced,
            patience=8,
            verbose=True,
        )

    def _cyclic_lr_scheduler(self):
        """Get the cyclic learning rate scheduler

        Returns:
            torch.optim: cyclic learning rate scheduler
        """
        return optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.model_config.learning_rate / 10,
            max_lr=self.model_config.learning_rate,
            mode="triangular2",
            step_size_up=self.len_train_loader * 10, # FIXME: self.model_config.step_size_up, self.len_train_loader * 5,
            cycle_momentum=False,
        )

    def _one_cycle_lr_scheduler(self):
        """Get the one cycle learning rate scheduler

        Returns:
            torch.optim: one cycle learning rate scheduler
        """
        total_steps = self.len_train_loader * self.model_config.number_of_epochs
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.model_config.learning_rate, total_steps=total_steps
        )
