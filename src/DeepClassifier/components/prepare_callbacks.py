"""This module contains the code for PrepareCallbacks."""

import os
import time
import tensorflow as tf

from pathlib import Path

from DeepClassifier.entities import PrepareCallbacksConfig
from DeepClassifier import logger


class PrepareCallbacks:
    def __init__(self, config: PrepareCallbacksConfig) -> None:
        """Inits PrepareCallbacks.

        Args:
            config (PrepareCallbacksConfig): The PrepareCallbacksConfig.
        """
        logger.info(">>>>>>>>>>>> PrepareCallbacks Log Started <<<<<<<<<<<<")
        self.config = config

    @property
    def _create_tb_callbacks(self) -> tf.keras.callbacks.Callback:
        """Creates and returns the TensorBoard callback.

        Returns:
            tf.keras.callbacks.Callback: TensorBoard callback.
        """
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        logger.info("Creating directory for the TensorBoard logs")
        tb_running_log_dir = Path(
            os.path.join(
                self.config.tensorboard_root_log_dir,
                f"tb_logs_at_time_{timestamp}",
            )
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_checkpoint_callbacks(self) -> tf.keras.callbacks.Callback:
        """Creates and returns the ModelCheckpoint callback.

        Returns:
            tf.keras.callbacks.Callback: ModelCheckpoint callback.
        """
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True,
        )

    def get_tb_and_checkpoint_callbacks(self) -> list:
        """Returns a list containing the TensorBoard and ModelCheckpoint callbacks.

        Returns:
            list: The list.
        """
        logger.info(
            f"""Tensorboard and Checkpoint callbacks: {[
                self._create_tb_callbacks,
                self._create_checkpoint_callbacks
            ]}"""
        )
        return [self._create_tb_callbacks, self._create_checkpoint_callbacks]
