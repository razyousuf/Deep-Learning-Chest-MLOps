# 5. Create Endpoints

from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline # handles the data ingestion pipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline # handles the base model preparation pipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline # handles the model training pipeline

# Define a human-readable name for pipeline stages
STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline() # Create pipeline object
    data_ingestion.main() # Run the main ingestion logic
    logger.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed\n\n <<<<<<<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f"===========================")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline() # Create pipeline object
    prepare_base_model.main() # Run the main preparation logic
    logger.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed\n\n <<<<<<<<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME = "Model Trainer"
try:
    logger.info(f"===========================")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    model_training = ModelTrainingPipeline() # Create pipeline object
    model_training.main() # Run the main training logic, i.e., train the model
    logger.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed\n\n <<<<<<<<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e