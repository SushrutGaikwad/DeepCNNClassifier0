"""This module contains the code for DataIngestion."""

import os
import urllib.request as request

from zipfile import ZipFile
from pathlib import Path
from tqdm import tqdm

from DeepClassifier.entities import DataIngestionConfig
from DeepClassifier import logger
from DeepClassifier.utils import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        """Inits DataIngestion.

        Args:
            config (DataIngestionConfig): The DataIngestionConfig.
        """
        logger.info(">>>>>>>>>>>> Data Ingestion Log Started <<<<<<<<<<<<")
        self.config = config

    def download_data_file(self) -> None:
        """Downloads the data file."""
        logger.info("Trying to download the data file")
        # Download only when the file is not already downloaded
        if not os.path.exists(self.config.zipped_data_file_path):
            logger.info("Data file is not already present. So, downloading it")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL, filename=self.config.zipped_data_file_path
            )
            logger.info(
                f"The file {filename} downloaded with the information:\n {headers}"
            )
        else:
            logger.info(
                f"Data file is already present with a size of {get_size(Path(self.config.zipped_data_file_path))}. Hence, not downloading it again"
            )

    def _get_updated_list_of_files(self, list_of_files: list) -> list:
        """Returns an updated list of files that include only those files which
        are needed for training.

        Args:
            list_of_files (list): The list of files to be updated.

        Returns:
            list: The updated list of files.
        """
        updated_list_of_files = []
        logger.info("Looping over all the files in the list of files")
        logger.info("Only considering image files of cats and dogs")
        for file in list_of_files:
            # Only get those files having an extension of 'jpg' and those that
            # are in either the 'Cat' or the 'Dog' directories
            if file.endswith(".jpg") and ("Cat" in file or "Dog" in file):
                updated_list_of_files.append(file)

        return updated_list_of_files

    def _preprocess(self, zf: ZipFile, file: Path, working_dir: Path) -> None:
        """Extracts a file from the zipped data file.

        Args:
            zf (ZipFile): The zip file of the data.
            file (str): The file (path) in the zip data to be extracted.
            working_dir (str): The directory in which the file is to be
                extracted.
        """
        # Creating the path of the file that is to be extracted
        target_file_path = Path(os.path.join(working_dir, file))

        # We extract the file only if it does not already exists
        if not os.path.exists(target_file_path):
            zf.extract(str(file), str(working_dir))

        # If the size of the extracted file is 0 KB, we delete it
        if os.path.getsize(target_file_path) == 0:
            logger.info(
                f"The file '{target_file_path}' has zero size. Hence, deleting it"
            )
            os.remove(target_file_path)

    def unzip_and_clean_data_file(self) -> None:
        """Unzips and cleans the data file."""
        logger.info(
            "Unzipping and cleaning the data files, i.e., removing the unwanted files"
        )
        with ZipFile(file=self.config.zipped_data_file_path, mode="r") as zf:
            # Getting the list of files in the downloaded zip file
            logger.info("Getting the list of files in the downloaded zip file")
            list_of_files = zf.namelist()

            # Updating the list of files to only include files that we want
            # for training
            logger.info("Updating the list of files")
            updated_list_of_files = self._get_updated_list_of_files(
                list_of_files=list_of_files
            )

            # Extracting the files
            logger.info("Extracting and clearning the files")
            for file in tqdm(updated_list_of_files):
                self._preprocess(
                    zf=zf, file=file, working_dir=self.config.unzipped_file_dir
                )
