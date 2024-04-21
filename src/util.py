import os
from pathlib import Path


def load_text_from_file(filename: str, directory: str = __file__) -> str:
    module_path = Path(os.path.realpath(directory)).parent
    file_path = module_path / filename
    with Path.open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_file_path(filename: str, directory: str) -> str:
    module_path = Path(os.path.realpath(directory)).parent
    return module_path / filename
