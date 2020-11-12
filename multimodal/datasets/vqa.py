# stdlib
import os
import urllib
import zipfile
import json
from itertools import combinations
from statistics import mean

# librairies
from appdirs import user_data_dir
from tqdm import tqdm
from torch.utils.data import Dataset

# own
from multimodal.features import get_features
from multimodal.datasets import vqa_utils


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
        tokenize_answers=True,
        top_answers=3000,
        tokenization=None,
    ):
        """
        dir_download: dir for the multimodal cache (data will be downloaded in a vqa2/ folder inside this directory
        features: which visual features should be used. Choices: coco-bottomup or coco-bottomup-36
        dir_features: if None, default to dir_download.
        split: in [train, val, test]
        """
        self.dir_download = dir_download
        if self.dir_download is None:
            self.dir_download = user_data_dir(appname="multimodal")
        self.split = split
        self.features = features
        self.dir_features = dir_features
        self.tokenize_answers = tokenize_answers
        self.top_answers = top_answers
        self.tokenization = tokenization
        if tokenization:
            import sentencepiece as spm

            self.tokenizer = spm.SentencePieceProcessor()

        self.dir_dataset = os.path.join(self.dir_download, self.name, self.split)
        os.makedirs(self.dir_dataset, exist_ok=True)

        # path download question
        filename = os.path.basename(self.url_questions[self.split])
        self.path_questions = os.path.join(self.dir_dataset, filename)

        # path download annotations
        filename = os.path.basename(self.url_annotations[self.split])
        self.path_original_annotations = os.path.join(self.dir_dataset, filename)

        # processed annotations contain answer_token and answer scores
        self.processed_dir = os.path.join(self.dir_dataset, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.path_annotations_processed = os.path.join(
            self.processed_dir, "annotations.json"
        )
        self.download()
        self.process()
        if self.features is not None:
            self.load_features()

        self.load()  # load questions and annotations

        # This dictionnary will be used for evaluation
        self.qid_to_annot = {a["question_id"]: a for a in self.annotations}

    def load_features(self):
        if self.split == "test":
            self.feats = get_features(
                self.features,
                split="test2015",
                dir_cache=self.dir_features,
            )
        else:
            self.feats = get_features(
                self.features, split="trainval2014", dir_cache=self.dir_features
            )

    def process(self):
        """Process answers to create answer tokens, and prepare score computation.
        This follows the official VQA evaluation score.
        """
        if os.path.exists(self.path_annotations_processed):
            return

        self.load_original_annotations()
        print("Processing annotations")
        # process punctuation
        print("\tPre-Processing punctuation")
        for annot in tqdm(self.annotations):
            for ansDic in annot["answers"]:
                ansDic["answer"] = vqa_utils.processPunctuation(ansDic["answer"])
        # process scores of every answer
        print("\tPre-Computing answer scores")
        for annot in tqdm(self.annotations):
            annot["scores"] = {}
            unique_answers = set([a["answer"] for a in annot["answers"]])
            for ans in unique_answers:
                scores = []
                # score is average of 9/10 answers
                for items in combinations(annot["answers"], 9):
                    matching_ans = [item for item in items if item["answer"] == ans]
                    score = min(1, float(len(matching_ans)) / 3)
                    scores.append(score)
                annot["scores"][ans] = mean(scores)

        with open(self.path_annotations_processed, "w") as f:
            json.dump(self.annotations, f)

    def load_original_annotations(self):
        with zipfile.ZipFile(self.path_original_annotations) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                self.annotations = json.load(f)["annotations"]

    def load(self):
        print("Loading questions")
        with zipfile.ZipFile(self.path_questions) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                self.questions = json.load(f)["questions"]

        print("Loading annotations")
        with open(self.path_annotations_processed) as f:
            self.annotations = json.load(f)

    def download(self):
        os.makedirs(os.path.join(self.dir_download, self.name), exist_ok=True)
        url_questions = self.url_questions[self.split]
        download_path = self.path_questions
        if not os.path.exists(download_path):
            print(f"Downloading questions at {url_questions} to {download_path}")
            urllib.request.urlretrieve(url_questions, download_path)
            urllib.request.urlretrieve(url_questions, download_path)
        url_annotations = self.url_annotations[self.split]
        download_path = self.path_original_annotations
        if not os.path.exists(download_path):
            print(f"Downloading annotations {url_annotations} to {download_path}")
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

    def evaluate(self, predictions):
        """
        predictions: list of answer_id
        """
        scores = {"overall": [], "yes/no": [], "number": [], "other": []}
        for p in predictions:
            qid = p["question_id"]
            ans = p["answer"]
            annot = self.qid_to_annot[qid]
            score = annot["scores"].get(ans, 0.0)  # default score is 0
            ans_type = annot["answer_type"]
            scores["overall"].append(score)
            scores[ans_type].append(score)
        return {key: mean(score_list) for key, score_list in scores.items()}


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
            self.features,
            split="trainval",
            dir_cache=self.dir_features,
        )

    def load(self):
        print("Loading questions")
        with open(self.path_questions) as f:
            self.questions = json.load(f)["questions"]

        print("Loading annotations")
        with open(self.path_annotations_processed) as f:
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
