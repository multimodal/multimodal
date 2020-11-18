# multimodal

A collection of multimodal (vision and language) datasets and visual features for deep learning research.

Currently it supports the following datasets: 
- VQA v1
- VQA v2
- VQA-CP v1
- VQA-CP v2

And the following features: 
- COCO Bottom-Up Top-Down features (10-100)
- COCO Bottom-Up Top-Down features (36)


## Usage

Available datasets are : 
```python
# Visual Question Answering
from multimodal.datasets import VQA, VQA2, VQACP, VQACP2

# COCO
from multimodal.datasets import COCO
```

Available features are:
```python
from multimodal.features import COCOBottomUpFeatures
```

### Word embeddings

Word embeddings are implemented as pytorch modules. Thus, they are trainable if needed, but can be freezed.

The following pretrained embeddings are available: 
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
## Example


To use the VQA v2 dataset


```python

from multimodal.datasets import VQA2
import torch

trainset = VQA2(features="bottomup", split="train")

print(trainset[0])

# load some deep learning model
model = ...

train_lodaer = torch.utils.data.DataLoader(trainset)

for batch in train_lodaer:
    features = batch["visual"]["features"]
    question = batch["question"]["question"]
    answers = batch["annotation"]["answers"]
    out = model(batch)
```

