"""Custom loss module."""
import torch
import numpy as np

def is_tensor(x_vector):
    """Check if x is a tensor.

    Args:
        x_vector (torch.Tensor or np.ndarray): input vector

    Returns:
        bool: True if x is a tensor, False otherwise
    """
    return isinstance(x_vector, (torch.Tensor, torch.nn.parameter.Parameter))

def is_numpy(x_vector):
    """Check if x is a numpy array.

    Args:
        x_vector (torch.Tensor or np.ndarray): input vector

    Returns:
        bool: True if x is a numpy array, False otherwise
    """
    return isinstance(x_vector, np.ndarray)

def to_tensor(x_vector, device=None):
    """Convert x to a tensor.

    Args:
        x_vector (torch.Tensor or np.ndarray): input vector
        device (torch.device): device to move the tensor to

    Returns:
        torch.Tensor: tensor
    """
    if is_tensor(x_vector):
        return x_vector
    return torch.tensor(x_vector, device=device).float()

def to_numpy(x_vector):
    """Convert x to a numpy array.
    If x is a tensor, detach it from the graph and move it to the cpu.

    Args:
        x_vector (torch.Tensor or np.ndarray): input vector

    Returns:
        np.ndarray: numpy array
    """
    if is_tensor(x_vector):
        return x_vector.detach().cpu().numpy()
    return x_vector


class LossFactory:
    """Factory class for losses."""
    @staticmethod
    def create_loss(config):
        """Create a loss.

        Args:
            config (ModelConfig): model config

        Returns:
            torch.nn.Module: loss function
        """

        if config.loss_type == "huber":
            return torch.nn.SmoothL1Loss(beta=config.beta, reduction=config.reduction)#(beta=beta_value, reduction='none')
        if config.loss_type == "logcosh":
            return LogCoshLoss()
        if config.loss_type == "quantile":
            return QuantileLoss(quantile=config.quantile)#(quantile=0.6)
        if config.loss_type == "pinball":
            return PinballLoss()
        if config.loss_type == "combinedhp":
            return CombinedHPLoss(delta=config.delta)#(delta=deltavalue)
        if config.loss_type == "combinedlp":
            return CombinedLPLoss()
        if config.loss_type == "rmse":
            return torch.sqrt(torch.nn.MSELoss() + 1e-6)
        if config.loss_type == "mse":
            return torch.nn.MSELoss(reduction=config.reduction)#(reduction='none')
        raise ValueError("Invalid loss type")

class LogCoshLoss(torch.nn.Module):
    """Log-cosh loss for quantile regression."""
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        """Compute the log-cosh loss.

        Args:
            predicted (torch.Tensor or np.ndarray): predicted values
            target (torch.Tensor or np.ndarray): target values
        Returns:
            torch.Tensor or np.ndarray: mean loss"""
        if is_tensor(predicted):
            return torch.mean(torch.log(torch.cosh(predicted - target)))
        if is_numpy(predicted):
            return np.mean(np.log(np.cosh(predicted - target)))
        raise ValueError("Input must be a tensor or numpy array")


class QuantileLoss(torch.nn.Module):
    """Quantile loss for quantile regression."""
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, predicted, target):
        """Compute the quantile loss.

        Args:
            predicted (torch.Tensor or np.ndarray): predicted values
            target (torch.Tensor or np.ndarray): target values
        Returns:
            torch.Tensor or np.ndarray: mean loss"""
        if is_tensor(predicted):
            error = predicted - target
            loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
            return torch.mean(loss)
        if is_numpy(predicted):
            error = predicted - target
            loss = np.maximum(self.quantile * error, (self.quantile - 1) * error)
            return np.mean(loss)
        raise ValueError("Input must be a tensor or numpy array")

# when we have multiple predictions
class PinballLoss(torch.nn.Module):
    """Pinball loss for quantile regression."""
    def __init__(self, quantile=None):
        super().__init__()
        self.quantiles = quantile if quantile is not None else [0.1, 0.5, 0.9]
    def forward(self, predicted, target, mask=None):
        """Compute the pinball loss.

        Args:
            predicted (torch.Tensor or np.ndarray): predicted values
            target (torch.Tensor or np.ndarray): target values
            mask (torch.Tensor or np.ndarray): mask to apply to the loss (optional. Default: None)

        Returns:
            torch.Tensor or np.ndarray: mean loss"""
        quantile1, quantile2, quantile3 = self.quantiles
        err1 = predicted[0, :, :] - target
        err2 = predicted[1, :, :] - target
        err3 = predicted[2, :, :] - target
        pinball1 = torch.max(quantile1 * err1, (quantile1 - 1) * err1)
        # apply the mask
        pinball2 = torch.max(quantile2 * err2, (quantile2 - 1) * err2)
        pinball3 = torch.max(quantile3 * err3, (quantile3 - 1) * err3)

        if mask is not None:
            mask = mask.unsqueeze(2)
            pinball1 = pinball1.masked_select(mask)
            pinball2 = pinball2.masked_select(mask)
            pinball3 = pinball3.masked_select(mask)
        eq_sum = pinball1 + pinball2 + pinball3
        #individual_quantile_losses = [torch.mean(eq1), torch.mean(eq2), torch.mean(eq3)]
        return torch.mean(eq_sum)#, individual_quantile_losses

    # TODO: function not working properly atm. Needs to be reworked
    def pinball_loss_numpy(self, predicted, target, quantiles=None):
        """Compute the pinball loss.
        """
        quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        losses = []
        for quantile in quantiles:
            error = target - predicted
            loss = np.mean(np.maximum((quantile - 1) * error, quantile * error))
            losses.append(loss)
        return np.mean(losses), losses


class CombinedHPLoss(torch.nn.Module):
    """Combined huberized pinball loss for quantile regression."""
    def __init__(self, quantiles=None, delta=0.5):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        self.delta = delta
    def forward(self, predicted, target, mask=None):
        """Compute the combined huberized quantile loss.

        Args:
            predicted (torch.Tensor or np.ndarray): predicted values
            target (torch.Tensor or np.ndarray): target values
            mask (torch.Tensor or np.ndarray): mask to apply to the loss (optional. Default: None)
        Returns:
            torch.Tensor or np.ndarray: mean loss
        """
        if is_tensor(predicted):
            quantile1, quantile2, quantile3 = self.quantiles
            err1 = predicted[0, :, :] - target
            err2 = predicted[1, :, :] - target
            err3 = predicted[2, :, :] - target
            eq1 = torch.max(quantile1 * err1, (quantile1 - 1) * err1)
            eq2 = torch.max(quantile2 * err2, (quantile2 - 1) * err2)
            eq3 = torch.max(quantile3 * err3, (quantile3 - 1) * err3)
            e1_huber = torch.nn.HuberLoss(delta=self.delta, reduction='none')(predicted[0, :, :], target)
            e2_huber = torch.nn.HuberLoss(delta=self.delta, reduction='none')(predicted[1, :, :], target)
            e3_huber = torch.nn.HuberLoss(delta=self.delta, reduction='none')(predicted[2, :, :], target)
            eq1_weighted = eq1 * e1_huber
            eq2_weighted = eq2 * e2_huber
            eq3_weighted = eq3 * e3_huber

            if mask is not None:
                mask = mask.unsqueeze(2)
                eq1_weighted = eq1_weighted.masked_select(mask)
                eq2_weighted = eq2_weighted.masked_select(mask)
                eq3_weighted = eq3_weighted.masked_select(mask)
            eq_sum = eq1_weighted + eq2_weighted + eq3_weighted
            #individual_quantile_losses = [torch.mean(eq1), torch.mean(eq2), torch.mean(eq3)]
            return torch.mean(eq_sum)
        if is_numpy(predicted):
            # TODO: check if this is correct
            losses = []
            for quantile in self.quantiles:
                huber_loss = np.where(np.abs(target - predicted) < self.delta, 0.5 * \
                                        np.square(target - predicted), self.delta * \
                                        (np.abs(target - predicted) - 0.5 * self.delta))
                quantile_weight = np.where((target - predicted) >= 0, quantile, (1 - quantile))
                weighted_loss = quantile_weight * huber_loss
                losses.append(np.mean(weighted_loss))
            mean_loss = np.mean(losses)
            return mean_loss, losses
        raise ValueError("Input must be a tensor or numpy array")

class CombinedLPLoss(torch.nn.Module):
    """Combined logcosh pinball loss for quantile regression."""
    def __init__(self, quantiles=None):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
    def forward(self, predicted, target):
        """Compute the combined logcosh quantile loss.

        Args:
            predicted (torch.Tensor or np.ndarray): predicted values
            target (torch.Tensor or np.ndarray): target values
        Returns:
            torch.Tensor or np.ndarray: mean loss
        """
        losses = []
        if is_tensor(predicted):
            for quantile in self.quantiles:
                log_cosh_loss_value = LogCoshLoss()(predicted, target)
                quantile_loss = torch.where((target - predicted) >= 0, quantile * log_cosh_loss_value, (1 - quantile) * log_cosh_loss_value)
                losses.append(quantile_loss.mean())
            mean_loss = torch.mean(torch.stack(losses))
            return mean_loss, losses
        if is_numpy(predicted):
            for quantile in self.quantiles:
                log_cosh_loss_value = LogCoshLoss()(predicted, target)
                quantile_loss = np.where((target - predicted) >= 0, quantile * log_cosh_loss_value, (1 - quantile) * log_cosh_loss_value)
                losses.append(np.mean(quantile_loss))
            mean_loss = np.mean(losses)
            return mean_loss, losses
        raise ValueError("Input must be a tensor or numpy array")


class HuberLoss():
    """Huber loss for quantile regression."""
    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, predicted, target):
        error = target - predicted
        abs_error = np.abs(error)
        squared_loss = 0.5 * np.square(error)
        linear_loss = self.delta * (abs_error - 0.5 * self.delta)
        return np.mean(np.where(abs_error <= self.delta, squared_loss, linear_loss))


class MSELoss():
    """Mean squared error loss."""
    def __init__(self):
        pass

    def __call__(self, predicted, target):
        return np.mean(np.square(predicted - target))
