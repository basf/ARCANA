
"""Early stopping class"""
from arcana.logger import logger
log = logger.get_logger("arcana.regularizations.stopping_criteria")


class EarlyStopping:
    """Early stopping class"""
    def __init__(self, criterion_rule='PQ', training_strip=10, alpha=0.1, patience=4):
        """Early stopping class

        Args:
            criterion_rule (str, optional): criterion to use. Defaults to 'PQ'.(GL and UP are the other options)
            training_strip (int, optional): training strip length. Defaults to 5.
            alpha (float, optional): alpha value for GL and PQ according to Prichelt et al. Defaults to 0.1.
            patience (int, optional): patience for early stopping. Defaults to 5.
        """
        self.criterion_rule = criterion_rule
        self.training_strip = training_strip
        self.alpha = alpha
        self.patience = patience
        self.counter = 0
        self.up_counter = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def step(self, train_loss, val_loss, epoch):
        """Step the early stopping by three criteria: GL, UP, PQ or PQUP
        - GL: generalization loss
        - UP: validation loss is increasing
        - PQ: progress quotient
        - PQUP: combination of PQ and UP

        Args:
            train_loss (float): training loss
            val_loss (float): validation loss
            epoch (int): epoch number

        Returns:
            bool: True if early stopping criteria is met
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.up_counter = 0
        else:
            criteria_met = False

            if self.criterion_rule == 'GL':
                # calculate the generalization loss
                generalization_loss = 100* ((val_loss / self.best_val_loss) - 1)
                if generalization_loss > self.alpha:
                    log.info(f"GL: {generalization_loss} > {self.alpha} @epoch {epoch} - up_counter {self.up_counter}")
                    criteria_met = True

            elif self.criterion_rule == 'PQ':
                # calculate the progress quotient (generalization loss / progress loss)
                if epoch % self.training_strip == 0:
                    p_k = 1000 *\
                        ((sum(self.train_losses[-self.training_strip:]) / (self.training_strip *\
                                                min(self.train_losses[-self.training_strip:]))) -1)
                    generalization_loss = 100 * ((val_loss / self.best_val_loss) - 1)
                    if generalization_loss/p_k > self.alpha:
                        #log.info(f"PQ: {gl/p_k} > {self.alpha} @epoch {epoch} - up_counter {self.up_counter}")
                        criteria_met = True

            elif self.criterion_rule == 'UP':
                if epoch > self.training_strip:
                    if val_loss > self.val_losses[-self.training_strip]:
                        #log.info(f"UP: {val_loss} > {self.val_losses[-self.training_strip]} @epoch {epoch} - up_counter {self.up_counter}")
                        criteria_met = True

            elif self.criterion_rule == 'PQUP':
                # calculate the progress quotient (generalization loss / progress loss)
                if epoch % self.training_strip == 0:
                    p_k = 1000 *\
                        ((sum(self.train_losses[-self.training_strip:]) / (self.training_strip *\
                                                min(self.train_losses[-self.training_strip:]))) -1)
                    generalization_loss = 100 * ((val_loss / self.best_val_loss) - 1)
                    if generalization_loss/p_k > self.alpha:
                        #log.info(f"PQ: {gl/p_k} > {self.alpha} @epoch {epoch} - up_counter {self.up_counter}")
                        criteria_met = True


                elif epoch > self.training_strip:
                    if val_loss > self.val_losses[-self.training_strip]:
                        #log.info(f"UP: {val_loss} > {self.val_losses[-self.training_strip]} @epoch {epoch} - up_counter {self.up_counter}")
                        criteria_met = True

            if criteria_met:
                self.up_counter += 1
                if self.up_counter > self.patience:
                    log.info(f"Early stopping @epoch {epoch}")
                    return True
            else:
                self.up_counter = 0
        return False
