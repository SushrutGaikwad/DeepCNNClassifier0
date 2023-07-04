from DeepClassifier.config import ConfigurationManager
from DeepClassifier.components import Evaluation
from DeepClassifier import logger


STAGE_NAME = "Evaluation"


def main():
    config = ConfigurationManager()
    evaluation_config = config.get_evaluation_config()
    evaluation = Evaluation(config=evaluation_config)
    evaluation.evaluation()
    evaluation.save_score()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} Stage Started <<<<<<<<<<<<")
        main()
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} Stage Completed <<<<<<<<<<<<\n\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
