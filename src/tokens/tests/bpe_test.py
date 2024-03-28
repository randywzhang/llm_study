import pytest

from util import load_text_from_file

from ..bpe import BasicBPETokenizer


@pytest.fixture()
def basic_bpe_tokenizer() -> BasicBPETokenizer:
    return BasicBPETokenizer()


class TestBasicBPETokenizer:
    def test_basic_bpe_tokenizer(self, basic_bpe_tokenizer: BasicBPETokenizer):
        text = load_text_from_file("test_text.txt", __file__)
        base_encoding = basic_bpe_tokenizer.encode(text)
        basic_bpe_tokenizer.train(text, 276)
        encoding = basic_bpe_tokenizer.encode(text)

        assert len(encoding) < len(base_encoding)
        assert basic_bpe_tokenizer.decode(encoding) == text
