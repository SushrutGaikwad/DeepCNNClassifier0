"""This module contains configuration entities for all the components."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path  # Directory where the artifacts of data ingestion will be
    # saved
    source_URL: str  # URL of the data
    zipped_data_file_path: Path  # Path of the downloaded zipped data file
    unzipped_file_dir: Path  # Directory of the unzipped data file
