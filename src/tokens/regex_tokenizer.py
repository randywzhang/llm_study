import regex
from regex import Pattern

from .bpe import BASE_VOCAB_SIZE, BasicBPETokenizer

"""
?i: - ignore case
'[sdmt]|ll|ve|re - matches 's 'd 'm 't 'll 've 're for contractions/possessive
[^\r\n\p{L}\p{N}] - NOT \r \n unicode letter, unicode number
?+ - one or more times
\p{L}+ - one or more unicode letters
\p{N}{1,3} - 1 to 3 unicode numbers
 ? - optional space
[^\s\p{L}\p{N}]++ - one or more characters that are NOT whitespace, letters, numbers
[\r\n]* - zero or more \r or \n
\s*[\r\n] - any whitespace followed by \r or \n
\s+(?!\S) - matches one or more whitespace characters up to but not including the space before a non-whitespace character (' x' would not match, '  x' would match ' ')
\s+ - matches one or more whitespace characters
"""
GPT_4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(BasicBPETokenizer):
    _pattern: Pattern

    def __init__(
        self,
        pattern: str = GPT_4_SPLIT_PATTERN,
        next_token_id: int = BASE_VOCAB_SIZE,
    ):
        self._pattern = regex.compile(pattern)
        super().__init__(next_token_id)

    def train(self, text: str, vocab_size: int, _verbose: bool = False) -> None:
        regex_split_tokens: list[list[int]] = [
            self._utf8_tokenization(split)
            for split in regex.findall(self.pattern, text)
        ]

        for _ in range(vocab_size - self.base_vocab_size):
            regex_split_tokens = self._create_new_token(regex_split_tokens)

    def _create_new_token(
        self,
        tokens: list[list[int]],
    ) -> list[list[int]]:
        # find pair to create new token out of
        bp_counts = self._get_stats_rt(tokens)
        top_pair = max(bp_counts, key=bp_counts.get)
        self.merges[top_pair] = self.next_token_id

        # increment next token id
        self.next_token_id += 1

        # added new token, vocab needs update
        self._vocab_update_flag = True

        # merge pair
        return self._merge_rt(tokens, self.next_token_id, top_pair)

    def _merge_rt(
        self,
        tokens: list[list[int]],
        token_id: int,
        pair: tuple[int, int],
    ) -> list[list[int]]:
        return [self._merge(split, token_id, pair) for split in tokens]

    def _get_stats_rt(
        self, regex_split_tokens: list[list[int]]
    ) -> dict[tuple[int, int], int]:
        """
        gets the count for each token pair occurring across all splits
        """
        pair_counts = {}
        for regex_split in regex_split_tokens:
            for pair in zip(regex_split, regex_split[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pair_counts

    @property
    def pattern(self) -> Pattern:
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: str) -> None:
        self._pattern = regex.compile(pattern)
