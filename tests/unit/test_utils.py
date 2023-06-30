import pytest
from pathlib import Path
from box import ConfigBox
from ensure.main import EnsureError

from DeepClassifier.utils import read_yaml


class Test_read_yaml:
    yaml_files = [
        "tests/data/empty.yaml",
        "tests/data/demo.yaml",
    ]

    def test_read_yaml_empty(self):
        with pytest.raises(ValueError):
            read_yaml(Path(self.yaml_files[0]))

    def test_read_yaml_return_type(self):
        response = read_yaml(Path(self.yaml_files[-1]))
        assert isinstance(response, ConfigBox)

    @pytest.mark.parametrize("yaml_file_path", yaml_files)
    def test_read_yaml_bad_type(self, yaml_file_path):
        with pytest.raises(EnsureError):
            read_yaml(yaml_file_path)
