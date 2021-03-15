from multimodal.datasets.coco import download
from multimodal import DEFAULT_DATA_DIR
import os
from torchtext.data.utils import get_tokenizer
import numpy as np
from collections import defaultdict
import pickle
from typing import List
import torch


class BasicTokenizer:
    """
    This class maps word tokens to token_ids.
    In case of unknown token ids, the 
    It will also pad the data.

    Args:
        tokens (list): Tokens to add in the dictionnary. Those can be tokens from pretrain vectors.
        sentences (list): List of sentences that need to be tokenized first, before building the vocab.
            Tokens from those sentences will be added to the vocabulary if they were not in it already.
        name (str): name which will be used to save the tokenizer. Use a different name when changing the tokens.
        pad_token (str): token used to pad the data.
        unk_token (str): token used for unknown words. The id is saved in the attribute unk_token_id.
        pad_side (str): either "left" or "right". The pad_token_id attribute will save the position.
        dir_data (str): directory to save multimodal data.
    """

    base_url = "https://webia.lip6.fr/~dancette/multimodal/tokenizers/{name}"

    def __init__(
        self,
        tokens: List[str] = [],
        sentences: List[str] = [],
        name: str = None,
        pad_token="<pad>",
        unk_token="<unk>",
        padding_side="right",
        dir_data: str = None,
    ):
        if dir_data is None:
            dir_data = DEFAULT_DATA_DIR
        self.dir = os.path.join(dir_data, "tokenizers")
        self.tokenizer = get_tokenizer("basic_english")

        os.makedirs(os.path.join(self.dir), exist_ok=True)

        if name is None:
            name = hash(
                (tuple(tokens), tuple(sentences), pad_token, unk_token, padding_side)
            )
            name = abs(name)  # positive hash, nicer filename (1 bit is lost).
            name = str(name)

        self.path = os.path.join(self.dir, name)

        if self.path is not None and os.path.exists(self.path):
            print(f"Loading VQATokenizer at {self.path}")
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
                    tokens = self.tokenize(s, replace_unk=False)
                    for t in tokens:
                        if t not in tokens_set:
                            self.tokens.append(t)
                            tokens_set.add(t)

            self.pad_token = pad_token
            self.unk_token = unk_token
            self.unk_token_id = len(self.tokens)  # last token
            self.token_id = defaultdict(lambda: self.unk_token_id)
            self.tokens.append(self.unk_token)

            if padding_side == "right":
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
        url = cls.base_url.format(name=name)
        path = download(url, os.path.join(dir_data, "tokenizers"))
        return BasicTokenizer(name=name)

    def tokenize(self, s, replace_unk=True, padding=True):
        """
        This function will return the tokenized representation of the input.
        Example: tokenize("Hello there") will return ["hello", "there"], assuming both words are in the vocabulary.
        
        In case a list of strings is given as input, this function will add padding tokens to ensure that all
        outputs have the same length.

        Args:
            s (str | List[str]): Either a string or a list of string, to be tokenized.
            keep_unk (bool): If true, then the tokenizes will not replace unknown words with the UNK token. Default: false
            padding (bool): Whether to add the padding token or not.
        """
        if type(s) == str:
            tokens = self.tokenizer(s)
            if replace_unk:
                tokens = [
                    t if t in self.token_to_id else self.unk_token for t in tokens
                ]
            return tokens
        elif type(s) == list:
            sentences = [self.tokenizer(sentence) for sentence in s]
            max_lengths = max(len(sent) for sent in sentences)
            # Padding
            if padding:
                sentences = [
                    sentence + [self.pad_token] * (max_lengths - len(sentence))
                    for sentence in sentences
                ]
            return sentences

    def convert_tokens_to_ids(self, tokens):
        """
        Converts tokenized representations 
        Args:
            tokens (list): List of string tokens that will be converted to their token ids.
            If a token is missing from the vocabulary, it will be converted to self.unk_token_id.
            Padding tokens will be converted to self.pad_token_id.
        """
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

    def __call__(self, s, tensor_type="np"):
        """
        This method calls tokenize(convert_tokens_to_ids(s))

        Args:
            s (list|str): text or list of texts to tokenize.
            tensor_type (str): either "pt" for pytorch or "np" for numpy array.
        """
        tokens = self.tokenize(s)
        token_ids = self.convert_tokens_to_ids(tokens)
        if tensor_type == "pt":
            return torch.tensor(token_ids)
        return token_ids

    def get_num_tokens(self):
        return len(self.tokens)
