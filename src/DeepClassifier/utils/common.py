import os
import yaml
import json
import joblib

from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from box.exceptions import BoxValueError
from typing import Any

from DeepClassifier import logger


@ensure_annotations
def read_yaml(yaml_file_path: Path) -> ConfigBox:
    """Reads and returns the content of a YAML file.

    Args:
        yaml_file_path (Path): Path of the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        e: Empty file.

    Returns:
        ConfigBox: The content of the YAML file.
    """
    try:
        with open(yaml_file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{yaml_file_path}' loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(paths_of_directories: list, verbose=True):
    """Creates directories using a given list of their paths.

    Args:
        paths_of_directories (list): A list containing the paths of the
            directories to be created.
        verbose (bool, optional): Whether to log the creation. Defaults to
            True.
    """
    for path in paths_of_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created the directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves JSON data to a JSON file.

    Args:
        path (Path): Path of the JSON file to save the JSON data into.
        data (dict): The data to be saved into the JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads data from a JSON file.

    Args:
        path (Path): Path of the JSON file to be loaded.

    Returns:
        ConfigBox: The content of the JSON file.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_binary_file(data: Any, path: Path):
    """Saves data into a binary file.

    Args:
        data (Any): The data to be saved into the binary file.
        path (Path): Path where the binary file is to be created.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_binary_file(path: Path) -> Any:
    """Loads data from a binary file.

    Args:
        path (Path): Path of the binary file.

    Returns:
        Any: Data stored in the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of a file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size of the file in KB.
    """
    size_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_kb} KB"
