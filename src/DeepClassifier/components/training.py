"""This module contains the code for Training."""

import tensorflow as tf

from pathlib import Path

from DeepClassifier.entities import TrainingConfig
from DeepClassifier import logger


class Training:
    def __init__(self, config: TrainingConfig) -> None:
        """Inits Training.

        Args:
            config (TrainingConfig): The TrainingConfig.
        """
        logger.info(">>>>>>>>>>>> Training Log Started <<<<<<<<<<<<")
        self.config = config

    def get_updated_base_model(self):
        """Loads the updated base model in the variable `self.update_base_model`,
        that was saved while preparing the base model.
        """
        # Loading the updated base model
        logger.info("Loading the updated base model")
        self.updated_base_model = tf.keras.models.load_model(
            filepath=self.config.updated_base_model_path
        )

    def train_val_generator(self):
        """Saves the training and validation generators in the variables
        `self.train_generator` and `self.validate_generator`.
        """
        # Initializing a dictionary for the kwargs to pass to `ImageDataGenerator`
        logger.info(
            "Initializing a dictionary for the kwargs to pass to `ImageDataGenerator`"
        )
        datagen_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=self.config.params_validation_split,
        )

        # Initializing a dictionary for the kwargs to pass to
        # `ImageDataGenerator.flow_from_directory`
        logger.info(
            "Initializing a dictionary for the kwargs to pass to `ImageDataGenerator.flow_from_directory`"
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        # Creating `ImageDataGenerator` for validation
        logger.info("Creating `ImageDataGenerator` for validation")
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

        # Creating validation_generator
        logger.info("Creating validation_generator")
        self.validation_generator = val_datagen.flow_from_directory(
            directory=self.config.training_data_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

        if self.config.params_augmentation:  # If `augmentation` is `True`
            # Creating `ImageDataGenerator` for training using augmentation
            logger.info("Creating `ImageDataGenerator` for training using augmentation")
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=self.config.params_rotation_range,
                horizontal_flip=self.config.params_horizontal_flip,
                width_shift_range=self.config.params_width_shift_range,
                height_shift_range=self.config.params_height_shift_range,
                shear_range=self.config.params_shear_range,
                zoom_range=self.config.params_zoom_range,
                **datagen_kwargs,
            )
        else:
            logger.info(
                "Creating `ImageDataGenerator` for training without using augmentation"
            )
            train_datagen = val_datagen

        # Creating train_generator
        logger.info("Creating train_generator")
        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.training_data_dir,
            subset="training",
            shuffle=True,
            **dataflow_kwargs,
        )

    def train_model(self, callbacks: list):
        """Trains and saves the model using a list of callbacks.

        Args:
            callbacks (list): The list of callbacks.
        """
        # Finding the steps_per_epoch number
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        logger.info(f"steps_per_epoch = {self.steps_per_epoch}")

        # Finding the validation_steps number
        self.validation_steps = (
            self.validation_generator.samples // self.validation_generator.batch_size
        )
        logger.info(f"validation_steps = {self.validation_steps}")

        # Training the updated model
        logger.info("Starting the training of the updated model")
        self.trained_model = self.updated_base_model.fit(
            x=self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.validation_generator,
            callbacks=callbacks,
        )
        logger.info("Training completed. Saving the trained model")

        self.save_model(
            model=self.trained_model,
            path=self.config.trained_model_path,
        )

    @staticmethod
    def save_model(model: tf.keras.Model, path: Path):
        """Saves a model to the given path.

        Args:
            model (tf.keras.Model): The model to be saved.
            path (Path): The path to save the model to.
        """
        # Saving the model
        logger.info(f"Saving the model to: {path}")
        model.save(path)
