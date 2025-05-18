# 3. Configuration Manager
import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig
                                                )

class ConfigurationManager:
    """
    Manages the retrieval and setup of configuration objects required for different pipeline components.
    Loads configurations from YAML files and ensures required directories are created.
    """

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        """
        Initializes the ConfigurationManager.

        Args:
            config_filepath (str): Path to the main config YAML file.
            params_filepath (str): Path to the parameters YAML file.
        """
        # Load YAML configuration files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Ensure the global artifacts root directory exists
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Constructs the configuration for the data ingestion stage.

        Returns:
            DataIngestionConfig: A data class containing all configuration values
            needed by the DataIngestion component.
        """
        # Extract data ingestion config section
        config = self.config.data_ingestion

        # Ensure the root directory for data ingestion exists
        create_directories([config.root_dir])

        # Package values into a structured DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Constructs the configuration for preparing the base model.

        Returns:
            PrepareBaseModelConfig: A data class containing all configuration values
            needed by the PrepareBaseModel component.
        """
        # Extract prepare base model config section
        config = self.config.prepare_base_model

        # Ensure the root directory for prepare base model exists
        create_directories([config.root_dir])

        # Package values into a structured PrepareBaseModelConfig object
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_base_model_path=config.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_classes=self.params.CLASSES,
            params_weights=self.params.WEIGHTS
        )

        return prepare_base_model_config


    def get_training_config(self) -> TrainingConfig:
        """
        Constructs the configuration for the training stage.
        Returns:
            TrainingConfig: A data class containing all configuration values
            needed by the Training component.
        """

        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chest-CT-Scan-data")
        create_directories([Path(training.root_dir)]) 
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.IS_AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )
        return training_config   