import os
from pathlib import Path

import pytest

from ..bpe import BasicBPETokenizer


def load_test_text(filename: str = "test_text.txt") -> str:
    module_path = Path(os.path.realpath(__file__)).parent
    file_path = module_path / filename
    with Path.open(file_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture()
def basic_bpe_tokenizer() -> BasicBPETokenizer:
    return BasicBPETokenizer()


class TestBasicBPETokenizer:
    def test_basic_bpe_tokenizer(self, basic_bpe_tokenizer: BasicBPETokenizer):
        text = load_test_text()
        base_encoding = basic_bpe_tokenizer.encode(text)
        basic_bpe_tokenizer.train(text, 276)
        encoding = basic_bpe_tokenizer.encode(text)

        assert len(encoding) < len(base_encoding)
        assert basic_bpe_tokenizer.decode(encoding) == text
