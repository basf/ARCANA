"""Teacher forcing scheduler class"""
import random
import math

from arcana.logger import logger
log = logger.get_logger("arcana.teacher_forcing")
class TeacherForcingScheduler:
    """ Teacher forcing scheduler class"""
    def __init__(self, num_epochs, epoch_division, seq_start, seq_end,
                 start_learning_ratio=0.5, end_learning_ratio=0.005,
                 decay_stride=0.3):
        """Teacher forcing scheduler class

        Args:
            num_epochs (int): number of epochs
            epoch_division (int): number of epoch divisions
            seq_start (int): start of the sequence
            seq_end (int): end of the sequence
            start_learning_ratio (float, optional): start learning ratio. Defaults to 0.5.
            end_learning_ratio (float, optional): end learning ratio. Defaults to 0.005.
            decay_stride (float, optional): decay stride. Defaults to 0.3.
        """
        self.num_epochs = num_epochs
        self.epoch_division = epoch_division
        self.seq_start = seq_start
        self.seq_end = seq_end
        self.start_learning_ratio = start_learning_ratio
        self.end_learning_ratio = end_learning_ratio
        self.decay_stride = decay_stride

    def get_ratio(self):
        """Get the teacher forcing ratio. It is implemented as a linear decay function which is
        defined as:
        teacher_forcing_ratio = initial_teacher_forcing_ratio * exp(-decay_parameter * epoch)
        and it is bounded between 0 and 1
        The higher teacher forcing ratio is used in the beginning of the training and it is
        gradually decreased to 0

        Returns:
            list: list of teacher forcing ratios
            available_seq: list of available sequences
        """
        # Get divisor and remainder
        stride = self.num_epochs // self.epoch_division
        stride_decay = self.decay_stride  # decay rate parameter

        total_stride = sum(stride * math.exp(-stride_decay * i) for i in range(self.epoch_division))
        stride_list = [int(round((stride * math.exp(-stride_decay * i) / total_stride) * self.num_epochs)) for i in range(self.epoch_division)]

        seq_step = int((self.seq_end - self.seq_start) / self.epoch_division)

        # Loop through the batch size and create a list of lists with a rand int between 15 and 15+seq_step
        available_seq = []
        teacher_forcing_ratio = []
        for i in range(self.epoch_division):
            sub_seq = []
            for _ in range(stride_list[i]):
                sub_seq.append(random.randint(self.seq_start+(seq_step*(i)), self.seq_start+(seq_step*(i+1))))
            available_seq.append(sub_seq)
            step_learning_ratio = (self.start_learning_ratio - self.end_learning_ratio) / (stride_list[i]+1e-8)
            teacher_forcing_ratio.append([self.start_learning_ratio - step_learning_ratio * i for i in range(stride_list[i])])

        # Combine the lists into one list
        available_seq = [item for sublist in available_seq for item in sublist]
        teacher_forcing_ratio = [item for sublist in teacher_forcing_ratio for item in sublist]

        # get the index of the teacher forcing ratio list
        sort_index_teacher_forcing_ratio = sorted(range(len(teacher_forcing_ratio)),
                                                  key=lambda k: teacher_forcing_ratio[k], reverse=True)
        # sort the teacher forcing ratio list and the available sequence list based on the teacher forcing ratio list
        available_seq = [available_seq[i] for i in sort_index_teacher_forcing_ratio]
        teacher_forcing_ratio = [teacher_forcing_ratio[i] for i in sort_index_teacher_forcing_ratio]

        # Post-processing step to adjust the length of available_seq
        actual_len = len(available_seq)
        if actual_len < self.num_epochs:
            log.warning(f"Padding available_seq to match num_epochs. Original length: {actual_len}, Target: {self.num_epochs}")
            padding_needed = self.num_epochs - actual_len
            # Pad with last value (or any other logic you find suitable)
            available_seq.extend([available_seq[-1]] * padding_needed)
            teacher_forcing_ratio.extend([teacher_forcing_ratio[-1]] * padding_needed)
        elif actual_len > self.num_epochs:
            log.warning(f"Trimming available_seq to match num_epochs. Original length: {actual_len}, Target: {self.num_epochs}")
            available_seq = available_seq[:self.num_epochs]
            teacher_forcing_ratio = teacher_forcing_ratio[:self.num_epochs]

        # Final check
        assert len(available_seq) == self.num_epochs, f"Final length mismatch: available_seq ({len(available_seq)}) vs num_epochs ({self.num_epochs})"

        log.info(f"Final length of available_seq: {len(available_seq)}, Final length of teacher_forcing_ratio: {len(teacher_forcing_ratio)}")

        return teacher_forcing_ratio, available_seq
