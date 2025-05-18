from dataclasses import dataclass # importing dataclass decorator which is used to create classes that are primarily used to store data without much functionality.
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration class for the Data Ingestion stage.
    This dataclass encapsulates all necessary paths and parameters return types required
    to perform data download and extraction.

    Attributes:
        root_dir (Path): Base directory for all data ingestion artifacts.
        source_URL (str): Google Drive shareable URL to the dataset zip file.
        local_data_file (Path): Path where the downloaded zip file will be saved.
        unzip_dir (Path): Directory where the zip file contents will be extracted.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """
    Configuration class for preparing the base model.
    This dataclass encapsulates all necessary paths and parameters required
    to prepare the base model.
    Attributes:
        root_dir (Path): Base directory for all model preparation artifacts.
        base_model_path (Path): Path to the pre-trained base model.
        updated_base_model_path (Path): Path where the updated base model will be saved.
        params_image_size (list): List of image dimensions for resizing.
        params_learning_rate (float): Learning rate for the model.
        params_include_top (bool): Flag to indicate if the top layer is included.
        params_classes (int): Number of classes for classification.
        params_weights (str): Weights initialization strategy.
    """
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_classes: int
    params_weights: str

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration class for the training stage.
    This dataclass encapsulates all necessary paths and parameters required
    to perform model training.
    Attributes:
        root_dir (Path): Base directory for all training artifacts.
        trained_model_path (Path): Path where the trained model will be saved.
        updated_base_model_path (Path): Path to the updated base model.
        training_data (Path): Directory containing the training data.
        params_epochs (int): Number of epochs for training.
        params_batch_size (int): Batch size for training.
        params_is_augmentation (bool): Flag to indicate if data augmentation is used.
        params_image_size (list): List of image dimensions for resizing.
    """
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list