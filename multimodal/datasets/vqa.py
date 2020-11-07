# stdlib
import os
import urllib

# librairies
from appdirs import user_data_dir

# pytorch
from torch.utils.data import Dataset
import zipfile
import json

from multimodal.features import get_features


class VQA(Dataset):

    name = "vqa"

    url_questions = {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip",
        "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip",
    }
    url_annotations = {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip",
    }

    def __init__(
        self,
        dir_download=None,
        features=None,
        dir_features=None,
        split="train",
        tokenization=None,
    ):
        """
        dir_download: dir for the multimodal.pytorch cache (data will be downloaded in a vqa2/ folder inside this directory
        features: which visual features should be used. Choices: coco-bottomup or coco-bottomup-36
        dir_features: if None, default to dir_download.
        split: in [train, val, test]
        """
        self.dir_download = dir_download
        if self.dir_download is None:
            self.dir_download = user_data_dir(appname="multimodal")
        self.features = features
        self.dir_features = dir_features
        self.split = split
        self.tokenization = tokenization
        if tokenization:
            import sentencepiece as spm
            self.tokenizer = spm.SentencePieceProcessor()

        # load questions
        self.download()
        self.load()
        if self.features is not None:
            self.load_features()

    def load_features(self):
        if self.split == "test":
            self.feats = get_features(
                self.features, split="test2015", dir_cache=self.dir_features,
            )
        else:
            self.feats = get_features(
                self.features, split="trainval2014", dir_cache=self.dir_features
            )

    def path_questions(self):
        url_questions = self.url_questions[self.split]
        filename = os.path.basename(url_questions)
        download_path = os.path.join(self.dir_download, self.name, filename)
        return download_path

    def path_annotations(self):
        url_annotation = self.url_annotations[self.split]
        filename = os.path.basename(url_annotation)
        download_path = os.path.join(self.dir_download, self.name, filename)
        return download_path

    def load(self):
        print("Loading questions")
        with zipfile.ZipFile(self.path_questions()) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                self.questions = json.load(f)["questions"]

        print("Loading annotations")
        with zipfile.ZipFile(self.path_annotations()) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                self.annotations = json.load(f)["annotations"]

    def download(self):
        os.makedirs(os.path.join(self.dir_download, self.name), exist_ok=True)
        url_questions = self.url_questions[self.split]
        download_path = self.path_questions()
        if not os.path.exists(download_path):
            print(f"Downloading {url_questions} to {download_path}")
            urllib.request.urlretrieve(url_questions, download_path)
        download_path = self.path_annotations()
        url_annotations = self.url_annotations[self.split]
        if not os.path.exists(download_path):
            print(f"Downloading {url_annotations} to {download_path}")
            urllib.request.urlretrieve(url_annotations, download_path)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        data = {
            "question": self.questions[index],
            "annotation": self.annotations[index],
        }

        if self.features is not None:
            image_id = data["question"]["image_id"]
            data["visual"] = self.feats[image_id]

        return data


class VQA2(VQA):

    name = "vqa2"

    url_questions = {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    }
    url_annotations = {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    }


class VQACP(VQA):

    name = "vqacp"

    url_questions = {
        "train": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_train_questions.json",
        "test": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_test_questions.json",
    }

    url_annotations = {
        "train": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_train_annotations.json",
        "test": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_test_annotations.json",
    }

    def load_features(self):
        self.feats = get_features(
            self.features, split="trainval", dir_cache=self.dir_features,
        )

    def load(self):
        print("Loading questions")
        with open(self.path_questions()) as f:
            self.questions = json.load(f)

        print("Loading annotations")
        with open(self.path_annotations()) as f:
            self.annotations = json.load(f)


class VQACP2(VQACP):

    name = "vqacp2"

    url_questions = {
        "train": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json",
        "test": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json",
    }

    url_annotations = {
        "train": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json",
        "test": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json",
    }
