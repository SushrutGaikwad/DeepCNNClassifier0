from DeepClassifier.config import ConfigurationManager
from DeepClassifier.components import DataIngestion
from DeepClassifier import logger


STAGE_NAME = "Data Ingestion"


def main():
    config = ConfigurationManager()

    data_ingestion_config = config.get_data_ingestion_config()

    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_data_file()
    data_ingestion.unzip_and_clean_data_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} Stage Started <<<<<<<<<<<<")
        main()
        logger.info(f">>>>>>>>>>>> {STAGE_NAME} Stage Completed <<<<<<<<<<<<\n\n\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
