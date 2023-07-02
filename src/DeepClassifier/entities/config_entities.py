"""This module contains configuration entities for all the components."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path  # Directory where the artifacts of data ingestion will be
    # saved
    source_URL: str  # URL of the data
    zipped_data_file_path: Path  # Path of the downloaded zipped data file
    unzipped_file_dir: Path  # Directory of the unzipped data file


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path  # Directory where the artifacts of `PrepareBaseModel` will
    # be saved
    base_model_path: Path  # Path where the base model will be saved
    updated_base_model_path: Path  # Path where the updated base model will be
    # saved
    params_image_size: list  # Value of the `image_size` parameter that
    # will be passed on as an argument to the `input_shape` parameter of the
    # model
    params_learning_rate: float  # Value of the `learning_rate` parameter
    params_include_top: bool  # Value of the `include_top` parameter
    params_weights: str  # Value of the `weights` parameter
    params_classes: int  # Value of the `classes` parameter that will be
    # passed as an argument to the `units` parameter of the final output layer


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path  # Directory where the artifacts of `PrepareCallbacksConfig`
    # will be saved
    tensorboard_root_log_dir: Path  # Directory where the tensorboard logs will
    # be saved
    checkpoint_model_filepath: Path  # Directory where the model checkpoint
    # will be saved


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path  # Directory where the artifacts of `TrainingConfig` will be
    # saved
    trained_model_path: Path  # Path where the trained model will be saved
    updated_base_model_path: Path  # Path where the updated base model will be
    # saved
    training_data_dir: Path  # Directory where the training data is saved
    params_epochs: int  # Value of the `epochs` parameter
    params_batch_size: int  # Value of the `batch_size` parameter
    params_augmentation: bool  # Whether to use augmentation on images during
    # training
    params_image_size: list  # Value of the `image_size` parameter
    params_validation_split: float  # Value of the `validation_split` parameter
    params_rotation_range: float  # Value of the `rotation_range` parameter for
    # data augmentation
    params_horizontal_flip: bool  # Value of the `horizontal_flip` parameter
    # for data augmentation
    params_width_shift_range: float  # Value of the `width_shift_range`
    # parameter for data augmentation
    params_height_shift_range: float  # Value of the `height_shift_range`
    # parameter for data augmentation
    params_shear_range: float  # Value of the `shear_range` parameter for data
    # augmentation
    params_zoom_range: float  # Value of the `zoom_range` parameter for data
    # augmentation
