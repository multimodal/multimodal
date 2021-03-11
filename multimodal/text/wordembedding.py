import os
import torch.nn as nn
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import pretrained_aliases

from multimodal import DEFAULT_DATA_DIR


"""
https://www.reddit.com/r/MachineLearning/comments/axkmi0/d_why_is_code_or_libraries_for_wordpiece/
https://github.com/bheinzerling/bpemb
https://github.com/google/sentencepiece
"""


def get_dim_from_name(name):
    return int(pretrained_aliases[name].keywords["dim"])


class PretrainedWordEmbedding(nn.Module):
    def __init__(
        self,
        name: str,
        tokens: list,
        freeze=True,
        max_tokens: int = None,
        unk_init=None,
        dir_data: str = None,
        padding_idx=None,
        **kwargs,
    ):
        """
        name: One of [charngram.100d, fasttext.en.300d, fasttext.simple.300d, 
            glove.42B.300d, glove.6B.100d, glove.6B.200d, glove.6B.300d, glove.6B.50d, 
            glove.840B.300d, glove.twitter.27B.100d, glove.twitter.27B.200d, 
            glove.twitter.27B.25d, glove.twitter.27B.50d]
        tokens: specify list of tokens to be used. If `None`, 
            load all tokens from word embedding (with `max_tokens`)
        max_tokens: if `tokens` is None, this*
        cache: path where word embeddings will be downloaded.
        """
        super().__init__()
        dir_data = dir_data or DEFAULT_DATA_DIR
        cache = os.path.join(dir_data, "word-embeddings")
        dim = get_dim_from_name(name)
        vocab = pretrained_aliases[name](
            cache=cache, unk_init=unk_init, max_vectors=max_tokens,
        )
        if tokens is None:
            tokens = vocab.itos

        self.embedding = nn.Embedding(
            len(tokens), dim, padding_idx=padding_idx, **kwargs
        )
        n_missing = 0
        for token_id, token in enumerate(tokens):
            if token not in vocab.itos:
                n_missing += 1
                # print(f"Missing token : {token}")
            self.embedding.weight.data[token_id] = vocab[token]
        print(f"Number of missing tokens: {n_missing} / {len(tokens)}")

        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)


class PretrainedWordEmbeddingWithTokenizer(nn.Module):
    """
    Word Embedding class.
    """

    def __init__(
        self, tokens: list, dim: int, freeze=False, compute_stats=False,
    ):
        """
        Arguments:
            name: name of Vectors in torchtext.vocab.Vectors.
                Can be
            tokens: list of strings containing all the tokens in the vocabulary
            compute_stats: Total number of tokens and total number of unknown tokens processed will be saved in the
                ```self.stats['unknown']``` and ```self.stats['total']``` attribute
        """
        super().__init__()
        tokens = list(sorted(tokens))  # to keep always the same order
        self.tokens = ["<pad>", "<unk>"] + tokens  # padding and unknown token
        self.tokens_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.vocab = None  # lazy loading
        self.embedding = nn.Embedding(len(self.tokens), dim, padding_idx=0)

        if freeze:
            self.embedding.weight.requires_grad = False

        self.stats = {"unknown": 0, "total": 0, "unk_words": set()}
        self.compute_stats = compute_stats

        # torchtext basic_english tokenizer better matches
        # word embedding tokens such as glove.
        self.tokenizer = get_tokenizer("basic_english")

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        tokens: list = None,
        freeze=True,
        max_tokens: int = None,
        unk_init=None,
        dir_data: str = None,
    ):
        """
        name: One of [charngram.100d, fasttext.en.300d, fasttext.simple.300d, 
            glove.42B.300d, glove.6B.100d, glove.6B.200d, glove.6B.300d, glove.6B.50d, 
            glove.840B.300d, glove.twitter.27B.100d, glove.twitter.27B.200d, 
            glove.twitter.27B.25d, glove.twitter.27B.50d]
        tokens: specify list of tokens to be used. If `None`, 
            load all tokens from word embedding (with `max_tokens`)
        max_tokens: if `tokens` is None, this*
        cache: path where word embeddings will be downloaded.
        """
        dir_data = dir_data or DEFAULT_DATA_DIR
        cache = os.path.join(dir_data, "word-embeddings")
        dim = get_dim_from_name(name)
        vocab = pretrained_aliases[name](
            cache=cache, unk_init=unk_init, max_vectors=max_tokens,
        )
        if tokens is None:
            tokens = vocab.itos
        embedding = cls(tokens, dim=dim, freeze=freeze)
        n_missing = 0
        for token in embedding.tokens:
            token_id = embedding.tokens_to_id[token]
            if token not in vocab.itos:
                n_missing += 1
                # print(f"Missing token : {token}")
            embedding.embedding.weight.data[token_id] = vocab[token]
        print(f"Number of missing tokens: {n_missing} / {len(tokens)}")
        return embedding

    def filter_and_pad(self, sentences):
        max_lengths = max(len(sent) for sent in sentences)
        filtered_sentences = []
        # Filter unknown tokens
        for sent in sentences:
            filtered_sent = []
            for token in sent:
                if token in self.tokens_to_id:
                    filtered_sent.append(token)
                else:
                    filtered_sent.append("<unk>")
                    self.log_unknown(token)
                self.log_one()
            filtered_sentences.append(filtered_sent)
        sentences = filtered_sentences

        # Padding
        sentences = [
            sentence + ["<pad>"] * (max_lengths - len(sentence))
            for sentence in sentences
        ]
        return sentences

    def forward(self, sentences, tokenized=False):
        """
        Arguments:
            sentences: list of sentences (batch)
        """
        if not tokenized:
            sentences = [self.tokenizer(sentence) for sentence in sentences]
        sentences = self.filter_and_pad(sentences)
        # torch.max()
        token_ids = torch.LongTensor(
            [[self.tokens_to_id[t] for t in sentence] for sentence in sentences]
        ).to(device=self.embedding.weight.device)
        return self.embedding(token_ids)

    def state_dict(self, *args, **kwargs):
        """
        Save token list with the word embedding.
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["tokens"] = self.tokens
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        tokens = state_dict.pop("tokens")
        self.tokens = tokens
        self.tokens_to_id = {token: i for i, token in enumerate(self.tokens)}
        super().load_state_dict(state_dict, strict=strict)

    def log_one(self):
        if self.compute_stats:
            self.stats["total"] += 1

    def log_unknown(self, token):
        if self.compute_stats:
            self.stats["unk_words"].add(token)
            self.stats["unknown"] += 1
