from DeepClassifier.config import ConfigurationManager
from DeepClassifier.components import PrepareCallbacks, Training
from DeepClassifier import logger


STAGE_NAME = "Training"


def main():
    config = ConfigurationManager()

    prepare_callbacks_config = config.get_prepare_callbacks_config()
    prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
    callbacks = prepare_callbacks.get_tb_and_checkpoint_callbacks()

    training_config = config.get_training_config()
    training = Training(config=training_config)
    training.get_updated_base_model()
    training.train_val_generator()
    training.train_model(callbacks=callbacks)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} Stage Started <<<<<<<<<<<<")
        main()
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} Stage Completed <<<<<<<<<<<<\n\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
