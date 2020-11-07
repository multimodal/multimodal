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

