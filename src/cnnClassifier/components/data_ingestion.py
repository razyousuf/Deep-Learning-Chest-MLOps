# 4. Update the Components
import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    Handles data ingestion tasks such as downloading datasets from a given URL 
    and extracting them to a specified directory.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion component.
        Args:
            config (DataIngestionConfig): Configuration dataclass containing paths and URLs required for data ingestion.
        """
        self.config = config

    def download_file(self) -> str:
        """
        Downloads a file from a Google Drive URL and saves it locally.
        Returns:
            str: Path to the downloaded zip file.
        Raises:
            Exception: If the download fails.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            # Ensure the target directory exists
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Parse file ID from Google Drive share link
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # Download the file using gdown
            gdown.download(prefix + file_id, zip_download_dir, quiet=False)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

            return zip_download_dir

        except Exception as e:
            logger.exception("Failed to download the dataset.")
            raise e

    def extract_zip_file(self) -> None:
        """
        Extracts the downloaded zip file to the specified unzip directory.
        Returns:
            None
        Raises:
            Exception: If extraction fails or zip file is invalid.
        """
        unzip_path = self.config.unzip_dir

        # Ensure the extraction path exists
        os.makedirs(unzip_path, exist_ok=True)

        # Extract contents of the zip file
        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
                logger.info(f"Extracted zip file to {unzip_path}")
        except zipfile.BadZipFile as e:
            logger.error("The downloaded file is not a valid zip file.")
            raise e
