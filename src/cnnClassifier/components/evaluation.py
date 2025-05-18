import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
import mlflow
import mlflow.keras
from urllib.parse import urlparse

from dotenv import load_dotenv
import os
import dagshub
from dagshub.auth import add_app_token


# Load environment variables from .env and authenticate with DagsHub
load_dotenv()
dagshub_token = os.getenv("DAGSHUB_TOKEN")
add_app_token(dagshub_token)

# MLflow tracking repository (my dagshub repo)
dagshub.init(repo_owner='razyousufi350', repo_name='Deep-Learning-Chest-MLOps', mlflow=True)



class Evaluation: # This class is responsible for evaluating the model.
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self): # This method is responsible for creating the validation data generator.

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model: # This method is responsible for loading the model.
        return tf.keras.models.load_model(path)
    

    def evaluation(self): # This method is responsible for evaluating the model.
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    
    def save_score(self): # This method is responsible for saving the score of the model.
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    

    def log_into_mlflow(self): # This method is responsible for logging the model metrics into MLflow.
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")

    