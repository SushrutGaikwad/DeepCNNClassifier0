import os
import logging

from pathlib import Path


logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")


package_name = "DeepClassifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/config/__init__.py",
    f"src/{package_name}/pipeline/__init__.py",
    f"src/{package_name}/entities/__init__.py",
    f"src/{package_name}/constants/__init__.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "configs/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "research/trials.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        logging.info(f"Creating the directory '{filedir}' for the file '{filename}'.")
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filename)) or (os.path.getsize(filepath) == 0):
        with open(file=filepath, mode="w") as f:
            logging.info(f"Creating an empty file '{filepath}'.")
            pass  # creates an empty file
    else:
        logging.info(f"The file '{filename}' already exists.")
