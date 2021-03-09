from multimodal.datasets.coco import download, download_and_unzip
from multimodal import DEFAULT_DATA_DIR
import os
from torchtext.data.utils import get_tokenizer
import numpy as np
from collections import defaultdict
import pickle
from typing import List


class VQAQuestionTokenizer:
    """
    This class maps word tokens to token_ids.
    In case of unknown token ids, the 
    It will also pad the data.

    Args:
        sentences (list): List of sentences that need to be tokenized first, before building the vocab.
        additional_tokens (list): Tokens to add in the dictionnary if they were not there in the first place.
            This can be used to add vocabulary from pretrained word vectors for instance.
        name (str): name which will be used to save the tokenizer. Use a different name when changing the tokens.
    """

    urls = {
        "vqa2": "https://webia.lip6.fr/~dancette/multimodal",
    }

    def __init__(
        self,
        tokens: List[str] = None,
        sentences: List[str] = None,
        name: str = None,
        pad_token="<pad>",
        unk_token="<unk>",
        padding_size="right",
        dir_data: str = None,
    ):
        self.tokenizer = get_tokenizer("basic_english")

        if dir_data is None:
            dir_data = DEFAULT_DATA_DIR
        os.makedirs(
            os.path.join(dir_data, "tokenizers", "vqa_tokenizer"), exist_ok=True
        )

        if name is None:
            # hash the vocab to create a unique name ?
            name = "VQATokenizer"

        self.path = os.path.join(dir_data, "tokenizers", "vqa_tokenizer", name)

        if self.path is not None and os.path.exists(self.path):
            print("Loading VQATokenizer")
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self.tokens = data["tokens"]
            self.pad_token = data["pad_token"]
            self.pad_token_id = data["pad_token_id"]
            self.unk_token = data["unk_token"]
            self.unk_token_id = data["unk_token_id"]
        else:
            # Create tokenizer from scratch
            if tokens is not None:
                self.tokens = tokens
            else:
                self.tokens = []

            if sentences is not None:
                tokens_set = set(self.tokens)
                for s in sentences:
                    tokens = self.tokenize(s)
                    for t in tokens:
                        if t not in tokens_set:
                            self.tokens.append(t)
                            tokens_set.add(t)

            self.pad_token = pad_token
            self.unk_token = unk_token
            self.unk_token_id = len(self.tokens)  # last token
            self.token_id = defaultdict(lambda: self.unk_token_id)
            self.tokens.append(self.unk_token)

            if padding_size == "right":
                self.pad_token_id = self.unk_token_id + 1
                self.tokens.append(self.pad_token)
            else:
                self.pad_token_id = 0
                self.tokens = [self.pad_token] + self.tokens

            data = {
                "tokens": self.tokens,
                "pad_token": self.pad_token,
                "pad_token_id": self.pad_token_id,
                "unk_token": self.unk_token,
                "unk_token_id": self.unk_token_id,
            }

            with open(self.path, "wb") as f:
                data = pickle.dump(data, f)

        self.token_to_id = {token: id for id, token in enumerate(self.tokens)}

    @classmethod
    def from_pretrained(cls, name, dir_data=None):
        dir_data = dir_data or DEFAULT_DATA_DIR
        path = download(
            cls.urls[name], os.path.join(dir_data, "tokenizers", "vqa_tokenizer", name)
        )
        return VQAQuestionTokenizer(path=path)

    def tokenize(self, s):
        if type(s) == str:
            return self.tokenizer(s)
        elif type(s) == list:
            sentences = [self.tokenizer(sentence) for sentence in s]
            max_lengths = max(len(sent) for sent in sentences)
            # Padding
            sentences = [
                sentence + [self.pad_token] * (max_lengths - len(sentence))
                for sentence in sentences
            ]
            return sentences

    def convert_tokens_to_ids(self, tokens):
        if type(tokens[0]) == str:
            print(tokens)
            return np.array(
                [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
            )
        elif type(tokens[0]) == list:
            token_ids = [
                [self.token_to_id.get(token, self.unk_token_id) for token in sentence]
                for sentence in tokens
            ]
            # Padding
            max_lengths = max(len(sent) for sent in tokens)
            token_ids = [
                tids + [self.pad_token_id] * (max_lengths - len(tids))
                for tids in token_ids
            ]
            return np.array(token_ids)

    def __call__(self, s):
        tokens = self.tokenize(s)
        return self.convert_tokens_to_ids(tokens)
