from multimodal.text import BasicTokenizer
import numpy as np
import tempfile


def test_vqa_tokenizer():
    with tempfile.TemporaryDirectory() as d:
        tokenizer = BasicTokenizer(
            tokens=["hello", ",", "this", "is", "a", "tokenizer"], dir_data=d
        )
        assert tokenizer.tokenize("Hello, this is me") == [
            "hello",
            ",",
            "this",
            "is",
            "<unk>",
        ]

        assert tokenizer.tokenize("Hello, this is me", replace_unk=False) == [
            "hello",
            ",",
            "this",
            "is",
            "me",
        ]

        assert tokenizer.tokens == [
            "hello",
            ",",
            "this",
            "is",
            "a",
            "tokenizer",
            "<unk>",
            "<pad>",
        ]
        assert np.all(tokenizer("Hello, this is me") == np.array([0, 1, 2, 3, 6]))

        assert tokenizer.tokenize(["Hello you", "Hi this me"]) == [
            ["hello", "you", "<pad>"],
            ["hi", "this", "me"],
        ]

        assert np.all(
            tokenizer(["Hello you", "Hi this me"]) == np.array([[0, 6, 7], [6, 2, 6],])
        )


def test_vqa_tokenizer_with_corups():
    corpus = ["Hello, this is a tokenizer", "this tokenizer"]
    with tempfile.TemporaryDirectory() as d:
        tokenizer = BasicTokenizer(sentences=corpus, dir_data=d)
        print(tokenizer.tokens)
        print(tokenizer.token_to_id)
        assert tokenizer.tokenize("Hello, this is me") == [
            "hello",
            ",",
            "this",
            "is",
            "<unk>",
        ]

        assert tokenizer.tokenize("Hello, this is me", replace_unk=False) == [
            "hello",
            ",",
            "this",
            "is",
            "me",
        ]

        assert tokenizer.tokens == [
            "hello",
            ",",
            "this",
            "is",
            "a",
            "tokenizer",
            "<unk>",
            "<pad>",
        ]
        assert np.all(tokenizer("Hello, this is me") == np.array([0, 1, 2, 3, 6]))

        assert tokenizer.tokenize(["Hello you", "Hi this me"]) == [
            ["hello", "you", "<pad>"],
            ["hi", "this", "me"],
        ]
        assert np.all(
            tokenizer(["Hello you", "Hi this me"]) == np.array([[0, 6, 7], [6, 2, 6],])
        )


def test_loading_tokenizer():
    corpus = ["Hello, this is a tokenizer", "this tokenizer"]
    with tempfile.TemporaryDirectory() as d:
        tokenizer = BasicTokenizer(sentences=corpus, dir_data=d, name="temp-tokenizer")
        tokens = tokenizer.tokens

        tokenizer = BasicTokenizer(dir_data=d, name="temp-tokenizer")
        tokens2 = tokenizer.tokens

        assert tokens == tokens2
