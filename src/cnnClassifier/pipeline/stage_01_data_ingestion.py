from cnnClassifier.config.configuration import ConfigurationManager # handles reading YAML configs and preparing config entities
from cnnClassifier.components.data_ingestion import DataIngestion # handles downloading and unzipping dataset
from cnnClassifier import logger

# Define a human-readable name for pipeline stages
STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    """
    Class to encapsulate the data ingestion workflow logic.

    Responsibilities:
    - Initialize configuration using ConfigurationManager
    - Get structured ingestion config
    - Download dataset
    - Unzip the dataset
    """

    def __init__(self):
        pass  # Constructor does not need to initialize any internal state

    def main(self):
        """
        Executes the main ingestion logic:
        - Reads configuration
        - Initializes DataIngestion component
        - Downloads and extracts dataset
        """
        config = ConfigurationManager()  # Load configs from YAML
        data_ingestion_config = config.get_data_ingestion_config()  # Get config entity for ingestion
        data_ingestion = DataIngestion(config=data_ingestion_config)  # Create component with config

        data_ingestion.download_file()  # Step 1: Download dataset
        data_ingestion.extract_zip_file()  # Step 2: Extract downloaded zip


# Run the pipeline only if this file is executed directly (Entry Point to run the pipeline)
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()  # Create pipeline object
        data_ingestion.main()  # Run the main ingestion logic
        logger.info(f">>>>>>>>>>>>>>>>>> Stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e 
