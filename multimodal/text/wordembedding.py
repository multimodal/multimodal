import torchtext
from torchtext.vocab import pretrained_aliases
import torch.nn as nn
import torch
import functools
import spacy
from torchtext.data.utils import get_tokenizer

"""
https://www.reddit.com/r/MachineLearning/comments/axkmi0/d_why_is_code_or_libraries_for_wordpiece/
https://github.com/bheinzerling/bpemb
https://github.com/google/sentencepiece
"""


class WordEmbedding(nn.Module):
    """
    Word Embedding class.
    """

    def __init__(
        self, tokens: list, dim: int, unk_init=None, freeze=False, compute_stats=False,
    ):
        """
        Arguments:
            name: name of Vectors in torchtext.vocab.Vectors.
                Can be
            tokens: list of strings containing all the tokens in the vocabulary
            unk_init: function to pass to torchtext.Vocab
            compute_stats: Total number of tokens and total number of unknown tokens processed will be saved in the
                ```self.stats['unknown']``` and ```self.stats['total']``` attribute
        """

        super().__init__()
        tokens = list(sorted(tokens))  # to keep always the same order
        self.tokens = ["<pad>", "<unk>"] + tokens  # padding and unknown token
        self.tokens_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.vocab = None  # lazy loading
        self.unk_init = unk_init
        self.embedding = nn.Embedding(len(self.tokens), dim, padding_idx=0)

        if freeze:
            self.embeddings.weights.requires_grad = False

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
        cache: str = None,
    ):
        """
        tokens: specify list of tokens to be used. If `None`, load all tokens from word embedding (with `max_tokens`)
        max_tokens: if `tokens` is None, this*
        cache: path where word embeddings will be downloaded.
        """
        dim = int(pretrained_aliases[name].keywords["dim"])
        vocab = pretrained_aliases[name](
            cache=cache, unk_init=unk_init, max_vectors=max_tokens,
        )
        if tokens is None:
            tokens = vocab.itos
        embedding = cls(tokens, dim=dim)
        for token in embedding.tokens:
            id = embedding.tokens_to_id[token]
            if token not in vocab.itos:
                print(f"Missing token : {token}")
            embedding.embedding.weight.data[id] = vocab[token]
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

    def forward(self, sentences):
        """
        Arguments:
            sentences: list of sentences (batch)
        """
        sentences = [self.tokenizer(sentence) for sentence in sentences]
        sentences = self.filter_and_pad(sentences)
        # torch.max()
        token_ids = torch.LongTensor(
            [[self.tokens_to_id[t] for t in sentence] for sentence in sentences]
        ).to(device=self.embedding.weight.device)
        return self.embedding(token_ids)

    def state_dict(self):
        """
        Save token list with the word embedding.
        """
        state_dict = super().state_dict()
        state_dict["tokens"] = self.tokens
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        tokens = state_dict.pop("tokens")
        self.tokens = tokens
        super().load_state_dict(state_dict, strict=strict)
        self.initialized = True

    def log_one(self):
        if self.compute_stats:
            self.stats["total"] += 1

    def log_unknown(self, token):
        if self.compute_stats:
            self.stats["unk_words"].add(token)
            self.stats["unknown"] += 1
