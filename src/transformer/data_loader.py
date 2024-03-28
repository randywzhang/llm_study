from tokens.regex_tokenizer import RegexTokenizer
from util import load_text_from_file

raw_text = load_text_from_file("training_data.txt", __file__)

TOKENIZER_VOCAB_SIZE = 500


def get_bespoke_tokenizer(
    raw_text: str,
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
) -> RegexTokenizer:
    """
    Note: If you did a split for training/test data, you must
    ensure that the training data has the same set of unique
    characters that the test data does. Otherwise you will
    either error out or get garbage back.
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

    regex_tokenizer.train(raw_text, vocab_size)

    return regex_tokenizer


tokenizer = get_bespoke_tokenizer(raw_text)

import jaxlib
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import Array

data = jnp.array(tokenizer.encode(raw_text))

# split data into train and validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

BATCH_SIZE = 4
BLOCK_SIZE = 8

key = jrand.PRNGKey(0)


def get_batch(train: bool = True) -> tuple[Array, Array]:
    data = train_data if train else val_data
    ix = jrand.randint(key, (BATCH_SIZE,), 0, len(data) - BLOCK_SIZE)
    x = jnp.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = jnp.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y
