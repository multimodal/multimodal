# multimodal

[![PyPI](https://img.shields.io/pypi/v/multimodal.svg)](https://pypi.python.org/pypi/multimodal/)
[![Documentation Status](https://readthedocs.org/projects/multimodal/badge/?version=latest)](https://multimodal.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/multimodal/week)](https://pepy.tech/project/multimodal) 
[![Join the chat at https://gitter.im/multimodal-learning/multimodal](https://badges.gitter.im/multimodal-learning/multimodal.svg)](https://gitter.im/multimodal-learning/multimodal?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A collection of multimodal (vision and language) datasets and visual features for deep learning research. See the [Documentation](https://multimodal.readthedocs.io/en/latest/).

**Visual Features**
Currently it supports the following visual features (downloaded automatically): 
- COCO [Bottom-Up Top-Down](https://github.com/peteanderson80/bottom-up-attention) features (10-100)
- COCO [Bottom-Up Top-Down](https://github.com/peteanderson80/bottom-up-attention) features (36)

**Datasets**
It also supports the following datasets, with their evaluation metric ([VQA evaluation metric](https://visualqa.org/evaluation.html)) 
- VQA v1
- VQA v2
- VQA-CP v1
- VQA-CP v2

- [CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/)

Note that when instanciating those datasets, large data might be downloaded. You can always specify the `dir_data` argument when instanciating, or you can set the environment variable `MULTIMODAL_DATA_DIR` so that all data always goes to the specified directory.

**Models**
Bottom-Up and Top-Down attention (UpDown)

**WordEmbeddings**
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

**VQA**

Available VQA datasets are VQA, VQA v2, VQA-CP, VQA-CP v2, and their associated [pytorch-lightinng](https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html) data modules.

You can run a simple evaluation of predictions using the following commands. 
Data will be downloaded and processed if necessary. Predictions must have the same format as the official VQA result format (see https://visualqa.org/evaluation.html).
```bash
# vqa 1.0
python -m multimodal vqa-eval -p <path/to/predictions> -s "val"
# vqa 2.0
python -m multimodal vqa2-eval -p <path/to/predictions> -s "val"
# vqa-cp 1.0
python -m multimodal vqacp-eval -p <path/to/predictions> -s "val"
# vqa-cp 2.0
python -m multimodal vqacp2-eval -p <path/to/predictions> -s "val"
```

To use the datasets for your training runs, use the following:

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
We also provide a pytorch_lightning datamodule, available here: `multimodal.datasets.lightning.VQADataModule` and similarly for other VQA datasets.
See documentation.

**CLEVR**

```python
from multimodal.datasets import CLEVR

# Warning, this will download a 18Gb file. 
# You can specify the multimodal data directory 
#   by providing the dir_data argument
clevr = CLEVR(split="train") 
```

### Pretrained Tokenizer and Word embeddings

Word embeddings are implemented as pytorch modules. Thus, they are trainable if needed, but can be freezed.

Pretrained embedding weights are downloaded with torchtext. The following pretrained embeddings are available: 
    charngram.100d, fasttext.en.300d, fasttext.simple.300d, glove.42B.300d, glove.6B.100d, glove.6B.200d, glove.6B.300d, glove.6B.50d, glove.840B.300d, glove.twitter.27B.100d, glove.twitter.27B.200d, glove.twitter.27B.25d, glove.twitter.27B.50d

Usage

```python
from multimodal.text import PretrainedWordEmbedding
from multimodal.text import BasicTokenizer

# tokenizer converts words to tokens, and to token_ids. Pretrained tokenizers 
# save token_ids from an existing vocabulary.
tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa")

# Pretrained word embedding, freezed. A list of tokens as input to initialize embeddings.
wemb = PretrainedWordEmbedding.from_pretrained("glove.840B.300d", tokens=tokenizer.tokens, freeze=True)

embeddings = wemb(tokenizer(["Inputs are batched, and padded. This is the first batch item", "This is the second batch item."]))
```


### Models

The Bottom-Up and Top-Down Attention for VQA model is implemented. 
To train, run `python multimodal/models/updown.py --dir-data <path_to_multimodal_data> --dir-exp logs/vqa2/updown`

It uses pytorch lightning, with the class `multimodal.models.updown.VQALightningModule`

You can check the code to see other parameters.

You can train the model manually:

```python
from multimodal.models import UpDownModel
from multimodal.datasets.import VQA2
from multimodal.text import BasicTokenizer
vqa_tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa2")

train_dataset = VQA(split="train", features="coco-bottomup", dir_data="/tmp")
train_loader = torch.utils.data.Dataloader(train_dataset, collate_fn = VQA.collate_fn)

updown = UpDownModel(num_ans=len(train_dataset.answers))

for batch in train_loader:
    batch["question_tokens"] = vqa_tokenizer(batch["question"])
    out = updown(batch)
    logits = out["logits"]
    loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
    loss.backward()
    optimizer.step()
```

Or train it with Pytorch Lightning:

```python
from multimodal.datasets.lightning import VQA2DataModule
from multimodal.models.lightning import VQALightningModule
from multimodal.text import BasicTokenizer
import pytorch_lightning as pl

tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa2")

vqa2 = VQA2DataModule(
    features="coco-bottomup-36",
    batch_size=512,
    num_workers=4,
)

vqa2.prepare_data()
num_ans = len(vqa2.num_ans)

updown = UpDownModel(
    num_ans=num_ans,
    tokens=tokenizer.tokens,  # to init word embeddings
)

lightningmodel = VQALightningModule(
    updown,
    train_dataset=vqa2.train_dataset,
    val_dataset=vqa2.val_dataset,
    tokenizer=tokenizer,
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=30,
    gradient_clip_val=0.25,
    default_root_dir="logs/updown",
)

trainer.fit(lightningmodel, datamodule=vqa2)
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


