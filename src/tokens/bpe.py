BASE_VOCAB_SIZE = 256  # ASCII, TODO: consider multi-byte utf-8 chars
BASE_VOCAB = {idx: bytes([idx]) for idx in range(256)}


class BasicBPETokenizer:
    _vocab: dict[int, bytes]
    _base_vocab: dict[int, bytes]  # defaults to utf-8 encoding for 0 - 255
    _vocab_update_flag: bool
    merges: dict[tuple[int, int], int]
    base_vocab_size: int
    next_token_id: int

    def __init__(
        self,
        base_vocab: dict[int, str] = BASE_VOCAB,
        base_vocab_size: int = BASE_VOCAB_SIZE,
    ) -> None:
        self._vocab = {}
        self._base_vocab = base_vocab
        self.next_token_id = base_vocab_size
        self.base_vocab_size = base_vocab_size
        self._vocab_update_flag = True
        self.merges = {}

    # TODO: logging
    def train(self, text: str, vocab_size: int, _verbose: bool = False) -> None:
        tokens = self._utf8_tokenization(text)

        for _ in range(vocab_size - self.base_vocab_size):
            tokens = self._create_new_token(tokens)

    def encode(self, text: str) -> list[int]:
        tokens = self._utf8_tokenization(text)
        if len(tokens) < 2:
            return tokens

        while len(tokens) > 1:
            pair_counts = self._get_stats(tokens)

            # get the pair that maps to the lowest index in self.merges
            # (need to do earlier merges before later ones)
            earliest_merged_pair = min(
                pair_counts, key=lambda pair: self.merges.get(pair, float("inf"))
            )
            if earliest_merged_pair not in self.merges:
                break  # no more possible merges

            tokens = self._merge(
                tokens=tokens,
                token_id=self.merges[earliest_merged_pair],
                pair=earliest_merged_pair,
            )

        return tokens

    def decode(self, encoding: list[int]) -> str:
        # decode utf8 to str, replacing errors with ascii symbol (in case llm spits out garbage)
        return b"".join(self.vocab[token] for token in encoding).decode(
            encoding="utf-8", errors="replace"
        )

    def _create_new_token(self, tokens: list[int]) -> list[int]:
        """
        merges the pair of tokens that occurs most frequently into
        a new token
        """
        # find pair to create new token out of
        bp_counts = self._get_stats(tokens)
        top_pair = max(bp_counts, key=bp_counts.get)
        self.merges[top_pair] = self.next_token_id

        # increment next token id
        self.next_token_id += 1

        # added new token, vocab needs update
        self._vocab_update_flag = True

        # merge pair
        return self._merge(tokens, self.next_token_id, top_pair)

    def _merge(
        self,
        tokens: list[int],
        token_id: int,
        pair: tuple[int, int],
    ) -> list[int]:
        idx = 0
        new_tokens = []
        while idx < len(tokens):
            if idx == len(tokens) - 1:
                new_tokens.append(tokens[-1])
                break

            if pair == (tokens[idx], tokens[idx + 1]):
                new_tokens.append(token_id)
                idx += 2
            else:
                new_tokens.append(tokens[idx])
                idx += 1
        return new_tokens

    def _get_stats(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        """
        gets the count for each token pair occurring in the tokens list
        """
        pair_counts = {}
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pair_counts

    def _utf8_tokenization(self, text: str) -> list[int]:
        """
        converts a string of text into a list of integers representing
        the byte codes of the characters
        """
        return list(map(int, text.encode(encoding="utf-8")))

    @property
    def vocab(self) -> dict[int, str]:
        """
        builds a mapping from token encodings to the strings they represent
        """
        if self._vocab_update_flag:
            vocab = self._base_vocab
            for (p0, p1), idx in self.merges.items():
                # works because we build the mappings in the order they were
                # created so that subsequent mappings can use previous ones
                vocab[idx] = vocab[p0] + vocab[p1]
            self._vocab = vocab

        self._vocab_update_flag = False
        return self._vocab

    @vocab.setter
    def vocab(self, tiktoken_mergeable_ranks: dict[bytes, int]) -> None:
        self._vocab = {v: k for k, v in tiktoken_mergeable_ranks.items()}
        self._vocab_update_flag = False
