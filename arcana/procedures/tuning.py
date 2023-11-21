''' This module contains the class for tuning the model. The tuning is done using the Optuna library.'''
import os
import shutil
import json
import time
import joblib
import optuna
from optuna.trial import TrialState


from arcana.logger import logger
from arcana.utils import utils
from arcana.procedures  import predicting


log = logger.get_logger("arcana.procedures.tuning")

class TuneProcedure:
    """This class is the main class for tuning the model. It contains some of the necessary functions for tuning the
    model. The class also contains all the parameters for the tuning of the model. It also contains the functions for
    saving the model parameters and the data splits. The Tuning is done using the Optuna library."""
    def __init__(self, procedure) -> None:
        self.procedure = procedure   #training.TrainProcedure()
        self.optuna_path = os.path.join(self.procedure.model_config.result_path, "optuna")
        self.study = None


    def tuning(self):
        """Run the tuning. First create the study, then optimize the model, then prune the bad trials
        """

        self.study = optuna.create_study(study_name=self.procedure.general_config.test_id, direction="minimize",
                        sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())

        # optimize the model
        # pylint: disable=unnecessary-lambda
        self.study.optimize(lambda trial: self._objective(trial),
                    n_trials=self.procedure.procedure_config.number_of_trials, catch=(RuntimeError, ValueError))
        # prune the bad trials
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        # complete the trials
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        # show the study statistics
        log.info("Study statistics: ")
        log.info(f"Number of finished trials: {len(self.study.trials)}")
        log.info(f"Number of pruned trials: {len(pruned_trials)}")
        log.info(f"Number of complete trials: {len(complete_trials)}")

        log.info("Best trial:")
        best_trial_study = self.study.best_trial

        log.info(f"Value: {best_trial_study.value}")
        log.info("Params:")
        for key, value in best_trial_study.params.items():
            log.info(f"{key}: {value}")

        # save the trial parameters
        self.save_tuning_params()

        # rename the trials
        self.rename_trials()


    def _objective(self, trial):
        """Objective function for the optimization
        Args:
            trial (optuna.trial.Trial): Trial object
        Returns:
            score (float): Score of the trial
        """

        trial_path = utils.create_dir(os.path.join(self.optuna_path, f"trial_{trial.number}"))
        self.procedure.model_config.result_path = trial_path
        utils.prepare_optuna_folder_structure(trial_path)
        shutil.copy2(self.procedure.model_config.path_to_config, os.path.join(trial_path, "config_files"))

        # overwrite the model parameters with the trial parameters
        self.procedure.model_config.read_tuning_conf(trial)

        # show the attributes of the tuning_config
        trial_config_dict = vars(self.procedure.model_config)
        log.info(f"Trial {trial.number} parameters: {trial_config_dict}")
        with open(os.path.join(trial_path, f"optuna_parameters_trial_{trial.number}.json"), "w", encoding='utf-8') as file:
            json.dump(trial_config_dict, file, default=utils.handle_tensor)

        # train the model
        self.procedure.training(trial=trial)
        # predict the model
        predicting.PredictProcedure(self.procedure).predicting(model_folder=trial_path)
        #self.procedure.predicting(model_folder=trial_path)

        #TODO check the parameters_all or just the losses
        score = self.procedure.seq_2_seq_trainer.parameters_all["losses"]["val_loss_epoch"][-1]

        return score

    def save_tuning_params(self):
        """Save the tuning parameters and the study
        """

        params_path = utils.create_dir(os.path.join(self.optuna_path, "tuning_parameters",
                                                            time.strftime("%Y%m%d-%H%M%S")))
        # get all the trials parameters
        self.study.trials_dataframe().to_csv(f"{params_path}/result.csv")
        # save the study
        joblib.dump(self.study, f"{params_path}/study.pkl")
        #FIXME this is not working , check for parallel coordinate plot
        # optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        # save_fig(tuning_save_path, "parallel_plot")
        optuna.visualization.matplotlib.plot_slice(self.study)
        utils.save_optuna_fig(params_path, "slice_plot")
        #FIXME this is not working , check for contour plot
        # optuna.visualization.matplotlib.plot_contour(study)
        # save_fig(tuning_save_path, "contour_plot")
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        utils.save_optuna_fig(params_path, "optimization_plot")
        optuna.visualization.matplotlib.plot_param_importances(self.study)
        utils.save_optuna_fig(params_path, "importance_plot")
        optuna.visualization.matplotlib.plot_intermediate_values(self.study)
        utils.save_optuna_fig(params_path, "intermediate_plot")


    def rename_trials(self):
        """Rename the trials folders according to their state.
           Pruned trials are deleted.
        """

        for i in self.study.trials_dataframe().index:
            if i == self.study.best_trial.number:
                # Append "best" to the best trial
                old_trial_path = os.path.join(self.optuna_path, f"trial_{self.study.best_trial.number}")
                new_trial_path = os.path.join(self.optuna_path, f"trial_{i}_BEST")

            else:
                old_trial_path = os.path.join(self.optuna_path, f"trial_{i}")
                state = self.study.trials_dataframe().loc[i,"state"]
                if state == "PRUNED":
                    shutil.rmtree(old_trial_path)
                    continue
                new_trial_path = os.path.join(self.optuna_path, f"trial_{i}_{state}")
            os.rename(old_trial_path, new_trial_path)
