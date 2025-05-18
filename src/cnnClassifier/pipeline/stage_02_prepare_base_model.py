from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelTrainingPipeline:
    """
    Class to encapsulate the base model preparation workflow logic.

    Responsibilities:
    - Initialize configuration using ConfigurationManager
    - Get structured prepare base model config
    - Prepare the base model
    """

    def __init__(self):
        pass  # Constructor does not need to initialize any internal state

    def main(self):
        """
        Executes the main preparation logic:
        - Reads configuration
        - Initializes PrepareBaseModel component
        - Prepares the base model
        """
        config = ConfigurationManager()  # Load configs from YAML
        prepare_base_model_config = config.get_prepare_base_model_config()  # Get config entity for prepare base model
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)  # Create component with config

        prepare_base_model.get_base_model()  # Step 1: Prepare the base model
        prepare_base_model.update_base_model() # Step 2: Update the base model with custom layers


# Run the pipeline only if this file is executed directly (Entry Point to run the pipeline)
if __name__ == "__main__":
    try:
        logger.info("==========================")
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<")
        obj = PrepareBaseModelTrainingPipeline()  # Create pipeline object
        obj.main()  # Run the main preparation logic
        logger.info(f">>>>>>>>>>>>>>>>>> Stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e