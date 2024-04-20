import pickle
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import jax.random as jrand
from jax import Array

from tokens.regex_tokenizer import SAVE_FILE, RegexTokenizer
from util import load_text_from_file

TOKENIZER_VOCAB_SIZE = 500


def get_bespoke_tokenizer(
    raw_text: str,
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    load_from_file: bool = False,
    save_file: str = SAVE_FILE,
    train_further: bool = False,
    save_tokenizer: bool = False,
) -> RegexTokenizer:
    """
    Note: If you did a split for training/test data, you must
    ensure that the training data has the same set of unique
    characters that the test data does. Otherwise you will
    either error out or get garbage back.

    Also, if you are loading from file, you must make sure that
    the raw_text input is the same text that the saved tokenizer
    was trained with.
    """
    unique_chars = sorted(list(set(raw_text)))

    def base_tokenizer(text: str) -> list[int]:
        stoi = {ch: i for i, ch in enumerate(unique_chars)}
        return [stoi[ch] for ch in text]

    base_vocab = {i: ch.encode("utf-8") for i, ch in enumerate(unique_chars)}

    regex_tokenizer = RegexTokenizer(
        base_vocab=base_vocab,
        base_vocab_size=len(unique_chars),
        base_tokenizer=base_tokenizer,
    )

    if load_from_file:
        regex_tokenizer.load(save_file)
        regex_tokenizer.base_tokenizer = base_tokenizer
        if not train_further:
            return regex_tokenizer

    regex_tokenizer.train(raw_text, vocab_size)

    if save_tokenizer:
        regex_tokenizer.save()

    return regex_tokenizer


BATCH_SIZE = 4
BLOCK_SIZE = 8

JRAND_SEED = 0


class TransformerDataLoader:
    data_file: str
    directory: str
    raw_text: str
    tokenizer: RegexTokenizer
    encoding: list[int]

    _data: Array
    _train_data: Array
    _val_data: Array

    batch_size: int
    block_size: int

    prng_key: Array

    # ruff: noqa: PLR0913
    def __init__(
        self,
        data_file: str,
        directory: Optional[str] = __file__,
        tokenizer: RegexTokenizer = None,
        encoding_save_file: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
        block_size: int = BLOCK_SIZE,
        jrand_seed: int = JRAND_SEED,
    ):
        self.data_file = data_file
        self.directory = directory
        self.load_raw_text()

        self.tokenizer = tokenizer
        self.encoding = None
        if encoding_save_file:
            self.encoding_save_file = encoding_save_file
        else:
            self.encoding_save_file = "_".join(
                [data_file.split(".")[0], "encoding.pkl"]
            )

        self._data = None
        self._train_data = None
        self._val_data = None

        self.batch_size = batch_size
        self.block_size = block_size

        self.prng_key = jrand.PRNGKey(jrand_seed)

    def load_raw_text(self) -> None:
        self.raw_text = load_text_from_file(self.data_file, self.directory)

    def generate_encoding(self, tokenizer: RegexTokenizer = None) -> None:
        if tokenizer:
            self.tokenizer = tokenizer

        self.encoding = self.tokenizer.encode(self.raw_text)

    def save_encoding(self, encoding_save_file: Optional[str] = None) -> None:
        if not encoding_save_file:
            encoding_save_file = self.encoding_save_file

        with Path.open(encoding_save_file, "wb") as f:
            f.write(pickle.dumps(self.encoding))

    def load_encoding(self, encoding_save_file: Optional[str] = None) -> bool:
        if not encoding_save_file:
            encoding_save_file = self.encoding_save_file

        try:
            with Path.open(encoding_save_file, "rb") as f:
                # ruff: noqa: S301
                self.encoding = pickle.loads(f.read())
        except FileNotFoundError:
            return False

        return True

    def clear_data(self) -> None:
        self._data = None
        self._train_data = None
        self._val_data = None

    def get_batch(self, train: bool = True) -> tuple[Array, Array]:
        data = self.train_data if train else self.val_data
        ix = jrand.randint(
            self.prng_key, (self.batch_size,), 0, len(data) - self.block_size
        )
        x = jnp.stack([data[i : i + self.block_size] for i in ix])
        y = jnp.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y

    @property
    def data(self) -> Array:
        if self._data is None:
            self._data = jnp.array(self.encoding)

        return self._data

    @property
    def train_data(self) -> Array:
        if self._train_data is None:
            self._train_data = self.data[: int(0.9 * len(self.data))]

        return self._train_data

    @property
    def val_data(self) -> Array:
        if self._val_data is None:
            self._val_data = self.data[int(0.9 * len(self.data)) :]

        return self._val_data


if __name__ == "__main__":
    test_file_name = "tiny_shakespeare.txt"
    data_loader = TransformerDataLoader(test_file_name)
    tokenizer = get_bespoke_tokenizer(
        raw_text=data_loader.raw_text,
        vocab_size=5000,
        load_from_file=True,
        train_further=False,
        save_tokenizer=True,
    )

    print("encoding file...")
    if not data_loader.load_encoding():
        data_loader.generate_encoding(tokenizer)

        data_loader.save_encoding()

    x, y = data_loader.get_batch()

    print(f"Inputs: {x}")
    print(f"Outputs: {y}")
