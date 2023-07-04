"""This module contains the code for Evaluation."""

import tensorflow as tf
from pathlib import Path

from DeepClassifier.entities import EvaluationConfig
from DeepClassifier.utils import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig) -> None:
        """Inits Evaluation.

        Args:
            config (EvaluationConfig): The EvaluationConfig.
        """
        self.config = config

    def _val_generator(self):
        """Creates validation generator for evaluation."""
        datagen_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=self.config.params_validation_split,
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

        self.validation_generator = val_datagen.flow_from_directory(
            directory=self.config.training_data_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    def evaluation(self):
        """Evaluates the model."""
        self.model = self.load_model(path=self.config.model_path)
        self._val_generator()
        self.scores = self.model.evaluate(self.validation_generator)

    def save_scores(self):
        """Saves the scores (loss and accuracy) of the evaluated model."""
        scores = {"loss": self.scores[0], "accuracy": self.scores[1]}
        save_json(path=Path("scores.json"), data=scores)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Loads model from a given path.

        Args:
            path (Path): The path of the model.

        Returns:
            tf.keras.Model: The model.
        """
        return tf.keras.models.load_model(path)
