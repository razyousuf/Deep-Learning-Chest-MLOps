from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training
from cnnClassifier import logger


STAGE_NAME = "Model Trainer"

class ModelTrainingPipeline:
    """
    Class to encapsulate the model training workflow logic.

    Responsibilities:
    - Initialize configuration using ConfigurationManager
    - Get structured training config
    - Train the model
    """

    def __init__(self):
        pass  # Constructor does not need to initialize any internal state

    def main(self):
        """
        Executes the main training logic:
        - Reads configuration
        - Initializes ModelTrainer component
        - Trains the model
        """
        config = ConfigurationManager()  # Load configs from YAML
        training_config = config.get_training_config()  # Get config entity for training
        training = Training(config=training_config)  # Create component with config
        training.get_base_model()
        training.train_valid_generator()
        training.train()# Step 1: Train the model
    

# Run the pipeline only if this file is executed directly (Entry Point to run the pipeline)
if __name__ == "__main__":
    try:
        logger.info(f"===========================")
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<")
        obj = ModelTrainingPipeline()  # Create pipeline object
        obj.main()  # Run the main training logic
        logger.info(f">>>>>>>>>>>>>>>>>> Stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e

