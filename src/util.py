import os
from pathlib import Path

OUTPUT_FOLDER = "output_files"


# TODO: Fix all instances of directory argument, change to script_location
# or something more accurate. directory is misleading
def load_text_from_file(filename: str, directory: str = __file__) -> str:
    module_path = Path(os.path.realpath(directory)).parent
    file_path = module_path / filename
    with Path.open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_file_path(filename: str, directory: str) -> str:
    module_path = Path(os.path.realpath(directory)).parent
    return module_path / OUTPUT_FOLDER / filename
