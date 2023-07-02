from DeepClassifier.config import ConfigurationManager
from DeepClassifier.components import PrepareBaseModel
from DeepClassifier import logger


STAGE_NAME = "Prepare Base Model"


def main():
    config = ConfigurationManager()

    prepare_base_model_config = config.get_prepare_base_model_config()

    prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
    prepare_base_model.create_and_save_base_model()
    prepare_base_model.update_base_model_to_full_model_and_save_it()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} Stage Started <<<<<<<<<<<<")
        main()
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} Stage Completed <<<<<<<<<<<<\n\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
