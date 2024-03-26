import os
from pathlib import Path

import pytest
from _pytest.config import Config

from ..bpe import BasicBPETokenizer


def pytest_configure(config: Config) -> None:
    pass


def load_test_text(filename: str = "test_text.txt") -> str:
    module_path = Path(os.path.realpath(__file__)).parent
    file_path = module_path / filename
    with Path.open(file_path, "r", encoding="utf-8") as f:
        return f.read()


class TestBasicBPETokenizer:
    def test_basic_bpe_tokenizer(self):
        tokenizer = BasicBPETokenizer()
        text = load_test_text()
        base_encoding = tokenizer.encode(text)
        tokenizer.train(text, 276)
        encoding = tokenizer.encode(text)

        assert len(encoding) < len(base_encoding)
        assert tokenizer.decode(encoding) == text
