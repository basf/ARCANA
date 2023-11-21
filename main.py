from arcana.logger import logger
from arcana.procedures  import training, tuning, predicting
from arcana.procedures import finetuning

logger.setup_applevel_logger(file_name="arcana_debug.log")
log = logger.get_logger("arcana.main")


def run_test():
    """Run the test

    """
    arcana_procedure = training.TrainProcedure()
    if arcana_procedure.procedure_config.naive_training:
        arcana_procedure.training()

    if arcana_procedure.procedure_config.transfer_learning:
        arcana_procedure = finetuning.FineTuning(tl_strategy="decoder")
        if not arcana_procedure.procedure_config.optuna_tuning:
            arcana_procedure.training()

    if arcana_procedure.procedure_config.optuna_tuning:
        tuning.TuneProcedure(procedure=arcana_procedure).tuning()

    if arcana_procedure.procedure_config.predicting:
        predicting.PredictProcedure(arcana_procedure).predicting()
        # predicting.PredictProcedure(arcana_procedure).predicting_all(fix_sequency_length=22, plot_prediction=True)


if __name__ == "__main__":
    run_test()