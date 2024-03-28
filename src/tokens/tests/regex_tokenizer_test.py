import os
from pathlib import Path

import pytest

from ...tokens import gpt4
from ..regex_tokenizer import RegexTokenizer


def load_test_text(filename: str = "test_text.txt") -> str:
    module_path = Path(os.path.realpath(__file__)).parent
    file_path = module_path / filename
    with Path.open(file_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture()
def regex_tokenizer() -> RegexTokenizer:
    return RegexTokenizer()


class TestRegexTokenizer:
    def test_regex_tokenizer(self, regex_tokenizer: RegexTokenizer):
        text = load_test_text()
        base_encoding = regex_tokenizer.encode(text)
        regex_tokenizer.train(text, 276)
        encoding = regex_tokenizer.encode(text)

        assert len(encoding) < len(base_encoding)
        assert regex_tokenizer.decode(encoding) == text

    def test_special_char(self, regex_tokenizer: RegexTokenizer):
        text = load_test_text("test_text_special_char.txt")
        regex_tokenizer.train(text, 300)
        encoding = regex_tokenizer.encode(text)
        decoding = regex_tokenizer.decode(encoding)

        assert decoding == text

    def test_gpt_4_equivalence(self, regex_tokenizer: RegexTokenizer):
        """
        This doesn't actually train the tokenizer since we don't have
        access to the training data for gpt-4. It simply loads the
        vocab (_mergeable_ranks) and checks that the bpe algorithm is
        correct.
        """
        text = load_test_text("test_text_large.txt")
        mergeable_ranks = gpt4.GPT_4_CL100K_BASE_ENCODING._mergeable_ranks
        regex_tokenizer.merges = gpt4.recover_merges(mergeable_ranks)
        regex_tokenizer.vocab = mergeable_ranks

        encoding = regex_tokenizer.encode(
            text, str_tokenizer=gpt4.gpt_4_base_str_tokenizer
        )
        gpt_encoding = gpt4.GPT_4_CL100K_BASE_ENCODING.encode(text)
        decoding = regex_tokenizer.decode(gpt_encoding)
        gpt_decoding = gpt4.GPT_4_CL100K_BASE_ENCODING.decode(encoding)

        assert encoding == gpt_encoding
