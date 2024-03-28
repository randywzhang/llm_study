from typing import Optional

import regex
from regex import Pattern

from .bpe import BASE_VOCAB, BASE_VOCAB_SIZE, BasicBPETokenizer
from .gpt4 import GPT_4_SPLIT_PATTERN


class RegexTokenizer(BasicBPETokenizer):
    _pattern: Pattern
    _base_tokenizer: callable

    def __init__(
        self,
        pattern: str = GPT_4_SPLIT_PATTERN,
        base_vocab: dict[int, str] = BASE_VOCAB,
        base_vocab_size: int = BASE_VOCAB_SIZE,
        base_tokenizer: Optional[callable] = None,
    ):
        self._pattern = regex.compile(pattern)
        if not base_tokenizer:
            base_tokenizer = self._utf8_tokenization
        self.base_tokenizer = base_tokenizer
        super().__init__(base_vocab=base_vocab, base_vocab_size=base_vocab_size)

    def train(self, text: str, vocab_size: int, _verbose: bool = False) -> None:
        regex_split_tokens: list[list[int]] = [
            self.base_tokenizer(split) for split in regex.findall(self.pattern, text)
        ]

        for _ in range(vocab_size - self.base_vocab_size):
            regex_split_tokens = self._create_new_token(regex_split_tokens)

    def encode(self, text: str) -> list[int]:
        regex_split_tokens: list[list[int]] = [
            self.base_tokenizer(split) for split in regex.findall(self.pattern, text)
        ]

        tokens: list[int] = []

        for split in regex_split_tokens:
            tokens.extend(self._encode_chunk(split))

        return tokens

    # this is the same as the basic encode() function
    def _encode_chunk(self, split: list[int]) -> list[int]:
        if len(split) < 2:
            return split

        i = 1
        while len(split) > 1:
            pair_counts = self._get_stats(split)

            # get the pair that maps to the lowest index in self.merges
            # (need to do earlier merges before later ones)
            earliest_merged_pair = min(
                pair_counts, key=lambda pair: self.merges.get(pair, float("inf"))
            )
            if earliest_merged_pair not in self.merges:
                break  # no more possible merges

            split = self._merge(
                tokens=split,
                token_id=self.merges[earliest_merged_pair],
                pair=earliest_merged_pair,
            )
            i += 1

        return split

    def _create_new_token(
        self,
        tokens: list[list[int]],
    ) -> list[list[int]]:
        # find pair to create new token out of
        bp_counts = self._get_stats_rt(tokens)
        if not bp_counts:
            # we have created tokens representing each split in
            # the input text

            # we can begin creating tokens representing "word" pairs
            # or simply return. For now, let's return for simplicity
            return tokens
        top_pair = max(bp_counts, key=bp_counts.get)
        self.merges[top_pair] = self.next_token_id

        # increment next token id
        self.next_token_id += 1

        # added new token, vocab needs update
        self._vocab_update_flag = True

        # merge pair
        return self._merge_rt(tokens, self.merges[top_pair], top_pair)

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

    @property
    def base_tokenizer(self) -> callable:
        if not self._base_tokenizer:
            self._base_tokenizer = self._utf8_tokenization
        return self._base_tokenizer

    @base_tokenizer.setter
    def base_tokenizer(self, tokenizer: callable) -> None:
        self._base_tokenizer = tokenizer
