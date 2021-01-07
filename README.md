# multimodal

[![PyPI](https://img.shields.io/pypi/v/multimodal.svg)](https://pypi.python.org/pypi/multimodal/)

A collection of multimodal (vision and language) datasets and visual features for deep learning research.

Currently it supports the following visual features (downloaded automatically): 
- COCO [Bottom-Up Top-Down](https://github.com/peteanderson80/bottom-up-attention) features (10-100)
- COCO [Bottom-Up Top-Down](https://github.com/peteanderson80/bottom-up-attention) features (36)

It also supports the following datasets, with their evaluation metric ([VQA evaluation metric](https://visualqa.org/evaluation.html)) 
- VQA v1
- VQA v2
- VQA-CP v1
- VQA-CP v2


And also word embeddings (either from scratch, or pretrained from torchtext, that can be fine-tuned).


## Simple Usage

To install the library, run `pip install multimodal`. It is supported for python 3.6 and 3.7.

### Visual Features

Available features are COCOBottomUpFeatures

```python
>>> from multimodal.features import COCOBottomUpFeatures
>>> bottomup = COCOBottomUpFeatures(features="trainval_36", dir_data="/tmp")
>>> image_id = 13455
>>> feats = bottomup[image_id]
>>> print(feats.keys())
['image_w', 'image_h', 'num_boxes', 'boxes', 'features']
>>> print(feats["features"].shape)  # numpy array
(36, 2048)
```

### Datasets

Available datasets are VQA, VQA v2, VQA-CP, VQA-CP v2

```python
# Visual Question Answering
from multimodal.datasets import VQA, VQA2, VQACP, VQACP2

dataset = VQA(split="train", features="coco-bottomup", dir_data="/tmp")
item = dataset[0]

dataloader = torch.utils.data.Dataloader(dataset, collate_fn = VQA.collate_fn)

for batch in dataloader:
    out = model(batch)
    # training code...
```

### Word embeddings

Word embeddings are implemented as pytorch modules. Thus, they are trainable if needed, but can be freezed.

Pretrained embedding weights are downloaded with torchtext. The following pretrained embeddings are available: 
    charngram.100d, fasttext.en.300d, fasttext.simple.300d, glove.42B.300d, glove.6B.100d, glove.6B.200d, glove.6B.300d, glove.6B.50d, glove.840B.300d, glove.twitter.27B.100d, glove.twitter.27B.200d, glove.twitter.27B.25d, glove.twitter.27B.50d

Usage

```python
from multimodal.text import WordEmbedding

# Pretrained word embedding, freezed.
wemb = WordEmbedding.from_pretrained("glove.840B.300d", freeze=True)

# Word embedding from scratch, and trainable.
wemb = Wordembedding(tokens, dim=50, freeze=False)

embeddings = wemb(["Inputs are batched, and padded. This is the first batch item", "This is the second batch item."])
```

### API 

#### Features

```python
features = COCOBottomUpFeatures(
    features="test2014_36",   # one of [trainval2014, trainval2014_36, test2014, test2014_36, test2015, test2015_36]
    dir_data=None             # directory for multimodal data. By default, in the application directory for multimodal.
)
```

Then, to get the features for a specific image: 
```python
feats = features[image_id]
```

The features have the following keys : 
```python
{
    "image_id": int,
    "image_w": int,
    "image_h" : int,
    "num_boxes": int
    "boxes": np.array(N, 4),
    "features": np.array(N, 2048),
}
```

#### Datasets
```python
# Datasets
dataset = VQA(
    dir_data=None,       # dir where multimodal data will be downloaded. Default is HOME/.multimodal
    features=None,       # which visual features should be used. Choices: coco-bottomup or coco-bottomup-36
    split="train",       # "train", "val" or "test"
    min_ans_occ=8,       # Minimum occurences to keep an answer.
    dir_features=None,   # Specific directory for features. By default, they will be located in dir_data/features.
    label="multilabel",  # "multilabel", or "best". This changes the shape of the ground truth label (class number for best, or tensor of scores for multilabel)
)
item = dataset[0]
```

The `item` will contain the following keys : 
```python
>>> print(item.keys())
{'image_id',
'question_id',
'question_type',
'question',                 # full question (not tokenized, tokenization is done in the WordEmbedding class)
'answer_type',              # yes/no, number or other
'multiple_choice_answer',
'answers',
'image_id',
'label',                    # either class label (if label="best") or target class scores (tensor of N classes).
'scores',                   # VQA scores for every answer
}
```



#### Word embeddings

```python
# Word embedding from scratch, and trainable.
wemb = Wordembedding(
    tokens,   # Token list. We recommend using torchtext basic_english tokenizer.
    dim=50,   # Dimension for word embeddings.
    freeze=False   # freeze=True means that word embeddings will be set with `requires_grad=False`. 
)



wemb = WordEmbedding.from_pretrained(
    name="glove.840B.300d", # embedding name (from torchtext)
    tokens,                 # tokens to load from the word embedding.
    max_tokens=None,        # if set to N, only the N most common tokens will be loaded.
    freeze=True,            # same parameter as default model. 
    dir_data=None,          # dir where data will be downloaded. Default is multimodal directory in apps dir.
)

# Forward pass
sentences = ["How many people are in the picture?", "What color is the car?"]
wemb(
    sentences, 
    tokenized=False  # set tokenized to True if sentence is already tokenized.
)

```

