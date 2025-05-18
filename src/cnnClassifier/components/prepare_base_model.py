import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import tensorflow as tf


class PrepareBaseModel:
    """
    Prepares the base model by downloading a pre-trained model, extracting it,
    and saving the model architecture and weights.
    """

    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the PrepareBaseModel component.

        Args:
            config (PrepareBaseModelConfig): Configuration dataclass containing paths and URLs required for model preparation.
        """
        self.config = config

    def get_base_model(self):
        """
        Downloads and prepares the base model.
        Returns:
            tf.keras.Model: The pre-trained base model.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares the full model by adding 2 custom dense layers (we have 2 classes of data) and compiling it.
        Args:
            model (tf.keras.Model): The pre-trained base model.
            classes (int): Number of output classes.
            freeze_all (bool): Whether to freeze all layers.
            freeze_till (int): Layer index to freeze till.
            learning_rate (float): Learning rate for the optimizer.
        Returns:
            tf.keras.Model: The full model ready for training.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        predictions = tf.keras.layers.Dense(
            units=classes,
            activation='softmax'
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=predictions
        )

        full_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model
    
    def update_base_model(self):
        """
        Updates the base model by preparing the full model and saving it in the artifacts folder.
        Returns:
            None
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the model architecture and weights to the specified path.
        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): The model to be saved.
        Returns:
            None
        """
        model.save(path)
