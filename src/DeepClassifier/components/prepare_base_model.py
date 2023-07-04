"""This module contains the code for PrepareBaseModel."""

import tensorflow as tf

from pathlib import Path

from DeepClassifier.entities import PrepareBaseModelConfig
from DeepClassifier import logger


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        """Inits PrepareBaseModel.

        Args:
            config (PrepareBaseModelConfig): The PrepareBaseModelConfig.
        """
        logger.info(">>>>>>>>>>>> PrepareBaseModel Log Started <<<<<<<<<<<<")
        self.config = config

    def create_and_save_base_model(self):
        """Creates the base model (VGG16) and saves it."""
        # Getting the VGG16 model as the base model
        logger.info("Getting the VGG16 model as the base model")
        self.base_model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )

        # Getting the path of the base model
        logger.info("Getting the path of the base model")
        base_model_path = self.config.base_model_path

        # Saving the base model
        self.save_model(model=self.base_model, path=base_model_path)

    @staticmethod
    def _prepare_full_model(
        base_model: tf.keras.Model,
        classes: int,
        freeze_all: bool,
        freeze_till: int,
        learning_rate: float,
    ) -> tf.keras.Model:
        """Prepares full model using the base model.

        Args:
            base_model (tf.keras.Model): The base model.
            classes (int): Number of output classes.
            freeze_all (bool): Whether to freeze all the layers in the base model.
            freeze_till (int): Number of layers (from the end) of the base model to
                be made trainable. This parameter is ignored if `freeze_all` is `True`.
            learning_rate (float): The learning rate for the full model.

        Returns:
            tf.keras.Model: The full model.
        """
        if freeze_all:  # if `freeze_all` is `True`
            # we freeze the weights of all the layers of the model
            logger.info("Freezing all layers of the base model")
            base_model.trainable = False

        # If `freeze_all` is `False` and `freeze_till` is not `None` and > 0
        elif (freeze_till is not None) and (freeze_till > 0):
            # we freeze the weights of all the layers except the last `freeze_till` layers
            logger.info(
                f"Freezing the weights of all the layers except the last {freeze_till} layers of the base model"
            )
            for layer in base_model.layers[:-freeze_till]:
                layer.trainable = False

        # Flattening the output of the above base model
        logger.info(
            "Flattening the output of the base model to create a flattened layer"
        )
        flatten_in = tf.keras.layers.Flatten()(base_model.output)

        # Creating the output layer of the full model
        logger.info("Creating the output layer of the full model")
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
        )(flatten_in)

        # Creating and compiling the full model
        logger.info("Creating and compiling the full model")
        full_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=prediction,
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        # Getting the summary of the full model
        logger.info("Getting the summary of the full model")
        full_model.summary()

        # Returning the full model
        return full_model

    def update_base_model_to_full_model_and_save_it(self):
        """Creates full model from the base model and saves it."""
        # Creating the full model
        logger.info("Creating the full model")
        self.full_model = self._prepare_full_model(
            base_model=self.base_model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )

        # Saving the full model
        logger.info("Saving the full model")
        self.save_model(
            model=self.full_model,
            path=self.config.updated_base_model_path,
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
