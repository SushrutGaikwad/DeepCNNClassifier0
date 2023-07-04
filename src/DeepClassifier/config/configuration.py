"""This module contains the code for ConfigurationManager."""

import os
from pathlib import Path

from DeepClassifier.entities import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig,
)
from DeepClassifier.utils import read_yaml, create_directories
from DeepClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
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
        logger.info("Getting the config info for data ingestion")
        config = self.config.data_ingestion

        # Creating the directory 'artifacts/data_ingestion'
        logger.info("Creating the directory 'artifacts/data_ingestion'")
        create_directories(paths_of_directories=[config.root_dir])

        # Creating and returning `DataIngestionConfig`
        logger.info("Creating DataIngestionConfig")
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            zipped_data_file_path=config.zipped_data_file_path,
            unzipped_file_dir=config.unzipped_file_dir,
        )
        logger.info(f"DataIngestionConfig: {data_ingestion_config}")
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """Creates and returns PrepareBaseModelConfig.

        Returns:
            PrepareBaseModelConfig: The PrepareBaseModelConfig.
        """
        # Getting the values in the `prepare_base_model` key of the config.yaml
        # file
        logger.info("Getting the config info for preparing the base model")
        config = self.config.prepare_base_model

        # Creating the directory 'artifacts/prepare_base_model'
        logger.info("Creating the directory 'artifacts/prepare_base_model'")
        create_directories(paths_of_directories=[config.root_dir])

        # Creating and returning `PrepareBaseModelConfig`
        logger.info("Creating PrepareBaseModelConfig")
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )
        logger.info(f"PrepareBaseModelConfig: {prepare_base_model_config}")
        return prepare_base_model_config

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        """Creates and returns PrepareCallbacksConfig.

        Returns:
            PrepareCallbacksConfig: The PrepareCallbacksConfig.
        """
        # Getting the values in the `prepare_callbacks` key of the config.yaml
        # file
        logger.info("Getting the config info for preparing callbacks")
        config = self.config.prepare_callbacks

        # Creating the directories 'artifacts/prepare_callbacks',
        # 'artifacts/prepare_callbacks/tensorboard_logs', and
        # 'artifacts/prepare_callbacks/checkpoint'
        logger.info(
            "Creating the directories for tensorboard logs and model checkpoint"
        )
        checkpoint_model_dir = os.path.dirname(Path(config.checkpoint_model_filepath))
        create_directories(
            paths_of_directories=[
                Path(config.root_dir),
                Path(config.tensorboard_root_log_dir),
                Path(checkpoint_model_dir),
            ]
        )

        # Creating and returning `PrepareCallbacksConfig`
        logger.info("Creating PrepareCallbacksConfig")
        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
        )
        logger.info(f"PrepareCallbacksConfig: {prepare_callbacks_config}")
        return prepare_callbacks_config

    def get_training_config(self) -> TrainingConfig:
        """Creates and returns TrainingConfig.

        Returns:
            TrainingConfig: The TrainingConfig.
        """
        # Getting the values in the `training` key of the config.yaml
        # file
        logger.info("Getting the config info for model training")
        config = self.config.training

        # Creating the directory 'artifacts/training'
        logger.info("Creating the directory 'artifacts/training'")
        create_directories(paths_of_directories=[Path(config.root_dir)])

        # Getting the directory of the training data from the 'data ingestion'
        # key of the config.yaml file
        logger.info(
            "Creating the path of the directory training data using the 'data ingestion' key of the config.yaml"
        )
        training_data_dir = os.path.join(
            self.config.data_ingestion.unzipped_file_dir,
            "PetImages",
        )

        # Creating and returning `TrainingConfig`
        logger.info("Creating TrainingConfig")
        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(
                self.config.prepare_base_model.updated_base_model_path
            ),
            training_data_dir=Path(training_data_dir),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_validation_split=self.params.VALIDATION_SPLIT,
            params_rotation_range=self.params.ROTATION_RANGE,
            params_horizontal_flip=self.params.HORIZONTAL_FLIP,
            params_width_shift_range=self.params.WIDTH_SHIFT_RANGE,
            params_height_shift_range=self.params.HEIGHT_SHIFT_RANGE,
            params_shear_range=self.params.SHEAR_RANGE,
            params_zoom_range=self.params.ZOOM_RANGE,
        )
        logger.info(f"TrainingConfig: {training_config}")
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        """Creates and returns EvaluationConfig.

        Returns:
            EvaluationConfig: EvaluationConfig
        """
        # Getting the directory of the training data from the 'data ingestion'
        # key of the config.yaml file
        logger.info(
            "Creating the path of the directory training data using the 'data ingestion' key of the config.yaml"
        )
        training_data_dir = os.path.join(
            self.config.data_ingestion.unzipped_file_dir,
            "PetImages",
        )

        # Creating and returning `EvaluationConfig`
        logger.info("Creating EvaluationConfig")
        evaluation_config = EvaluationConfig(
            model_path=self.config.training.trained_model_path,
            training_data_dir=training_data_dir,
            params_validation_split=self.params.VALIDATION_SPLIT,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
        logger.info(f"EvaluationConfig: {evaluation_config}")
        return evaluation_config
