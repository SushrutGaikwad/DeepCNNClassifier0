"""This module contains the code for ConfigurationManager."""

from DeepClassifier.entities import DataIngestionConfig
from DeepClassifier.utils import read_yaml, create_directories
from DeepClassifier.constants import *
from DeepClassifier import logger


class ConfigurationManager:
    def __init__(
        self,
        config_file_path: Path = CONFIG_FILE_PATH,
        params_file_path: Path = PARAMS_FILE_PATH,
    ) -> None:
        """Inits ConfigurationManager.

        Args:
            config_file_path (Path, optional): Path of the config.yaml file.
                Defaults to the constant CONFIG_FILE_PATH.
            params_file_path (Path, optional): Path of the params.yaml file.
                Defaults to the constant PARAMS_FILE_PATH.
        """
        # Getting information in the config.yaml and params.yaml file
        self.config = read_yaml(yaml_file_path=config_file_path)
        self.params = read_yaml(yaml_file_path=params_file_path)

        # Creating the 'artifacts' directory
        create_directories(paths_of_directories=[self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Creates and returns DataIngestionConfig.

        Returns:
            DataIngestionConfig: The DataIngestionConfig.
        """
        # Getting the values in the `data_ingestion` key of the config.yaml
        # file
        logger.info(f"Getting the config info for data ingestion")
        config = self.config.data_ingestion

        # Creating the directory 'artifacts/data_ingestion'
        logger.info(f"Creating the directory 'artifacts/data_ingestion'")
        create_directories(paths_of_directories=[config.root_dir])

        # Creating and returning `DataIngestionConfig`
        logger.info(f"Creating DataIngestionConfig")
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            zipped_data_file_path=config.zipped_data_file_path,
            unzipped_file_dir=config.unzipped_file_dir,
        )
        logger.info(f"DataIngestionConfig: {data_ingestion_config}")
        return data_ingestion_config
