##########################################
# File: test_tokenizer.py                #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import textwrap

import pytest

from workshop.experiments.transformer_sequence_classification.tokenizer import Tokenizer, TokenizerConfig


def test_tokenizer():
    tokenizer = Tokenizer(config=TokenizerConfig())
    for tokens, expected_token_ids in [
        ("", [0]),
        ("a", [0, 2]),
        ("a{2}", [0] + [2] * 2),
        ("a{10}b{1}", [0] + [2] * 10 + [3] * 1),
        ("{2}", [0] * 2),
    ]:
        token_ids = tokenizer.encode(tokens)
        assert token_ids == expected_token_ids

    with pytest.raises(ValueError) as exc_info:
        tokenizer.encode("ax")
    assert str(exc_info.value) == textwrap.dedent(
        """\
        unsupported token
        tokens = 'ax'
        i = 1
        c = 'x'"""
    )

    with pytest.raises(ValueError) as exc_info:
        tokenizer.encode("a{0}")
    assert str(exc_info.value) == textwrap.dedent(
        """\
        repeat < 1
        tokens = 'a{0}'
        repeat = 0"""
    )

    with pytest.raises(ValueError) as exc_info:
        tokenizer.encode("a{1")
    assert str(exc_info.value) == textwrap.dedent(
        """\
        missing }
        tokens = 'a{1'"""
    )
