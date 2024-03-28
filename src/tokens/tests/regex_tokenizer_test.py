import pytest

from util import load_text_from_file

from ...tokens import gpt4
from ..regex_tokenizer import RegexTokenizer


@pytest.fixture()
def regex_tokenizer() -> RegexTokenizer:
    return RegexTokenizer()


class TestRegexTokenizer:
    def test_regex_tokenizer(self, regex_tokenizer: RegexTokenizer):
        text = load_text_from_file("test_text.txt", __file__)
        base_encoding = regex_tokenizer.encode(text)
        regex_tokenizer.train(text, 276)
        encoding = regex_tokenizer.encode(text)

        assert len(encoding) < len(base_encoding)
        assert regex_tokenizer.decode(encoding) == text

    def test_special_char(self, regex_tokenizer: RegexTokenizer):
        text = load_text_from_file("test_text_special_char.txt", __file__)
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
        text = load_text_from_file("test_text_large.txt", __file__)
        mergeable_ranks = gpt4.GPT_4_CL100K_BASE_ENCODING._mergeable_ranks
        regex_tokenizer.merges = gpt4.recover_merges(mergeable_ranks)
        regex_tokenizer.vocab = mergeable_ranks
        regex_tokenizer.base_tokenizer = gpt4.gpt_4_base_str_tokenizer

        encoding = regex_tokenizer.encode(text)
        gpt_encoding = gpt4.GPT_4_CL100K_BASE_ENCODING.encode(text)
        decoding = regex_tokenizer.decode(gpt_encoding)

        assert encoding == gpt_encoding
        assert decoding == text
