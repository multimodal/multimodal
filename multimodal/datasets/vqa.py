# stdlib
from multimodal.text.wordembedding import get_dim_from_name
import os
import json
from itertools import combinations
from statistics import mean
from collections import Counter
from typing import List

# librairies
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader, default_collate
from torchtext.data.utils import get_tokenizer
import pytorch_lightning as pl

# own
from multimodal.features import get_features
from multimodal.datasets import vqa_utils
from multimodal import DEFAULT_DATA_DIR
from multimodal.utils import download_and_unzip
from multimodal.datasets.vqa_utils import EvalAIAnswerProcessor
from multimodal.text import WordEmbedding


class AbstractVQA(Dataset):
    def get_all_tokens(self) -> List:
        raise NotImplementedError()

    def evaluate(self, predictions) -> float:
        raise NotImplementedError()


class VQA(AbstractVQA):

    SPLITS = ["train", "val", "test", "test-dev"]

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

    filename_questions = {
        "train": "OpenEnded_mscoco_train2014_questions.json",
        "val": "OpenEnded_mscoco_val2014_questions.json",
        "test": "OpenEnded_mscoco_test2015_questions.json",
        "test-dev": "OpenEnded_mscoco_test-dev2015_questions.json",
    }

    filename_annotations = {
        "train": "mscoco_train2014_annotations.json",
        "val": "mscoco_val2014_annotations.json",
    }

    def __init__(
        self,
        dir_data=None,
        features=None,
        split="train",
        min_ans_occ=8,
        dir_features=None,
        label="multilabel",
        tokenize_questions=False,
    ):
        """
        dir_data: dir for the multimodal cache (data will be downloaded in a vqa2/ folder inside this directory
        features: which visual features should be used. Choices: coco-bottomup or coco-bottomup-36
        split: in [train, val, test]
        dir_features: directory to download features. If None, defaults to $dir_data/features
        label: either `multilabel`, or `best`. For `multilabel`, GT scores for questions are
            given by the score they are assigned by the VQA evaluation. 
            If `best`, GT is the label of the top answer.
        tokenize_questions: If True, preprocessing will tokenize questions into tokens.
            The tokens are stored in item["question_tokens"].
        """
        self.dir_data = dir_data
        if self.dir_data is None:
            self.dir_data = DEFAULT_DATA_DIR
        self.split = split
        self.features = features
        self.dir_features = dir_features or os.path.join(self.dir_data)
        self.label = label
        self.min_ans_occ = min_ans_occ
        self.tokenize_questions = tokenize_questions
        if self.tokenize_questions:
            self.tokenizer = get_tokenizer("basic_english")

        # Test split has no annotations.
        self.has_annotations = self.split in self.url_annotations

        self.dir_dataset = os.path.join(self.dir_data, "datasets", self.name)

        self.dir_splits = {
            split: os.path.join(self.dir_dataset, split)
            for split in self.filename_questions.keys()
        }  # vqa2/train/

        self.dir_splits["test-dev"] = "test"  # same directory

        for s in self.dir_splits:
            os.makedirs(self.dir_splits[s], exist_ok=True)

        # path download question
        self.path_questions = {
            split: os.path.join(self.dir_splits[split], self.filename_questions[split])
            for split in self.filename_questions.keys()
        }  # vqa2/train/OpenEnded_mscoco_train2014_questions.json

        # path download annotations
        self.path_original_annotations = {
            split: os.path.join(
                self.dir_splits[split], self.filename_annotations[split]
            )
            for split in self.filename_annotations.keys()
        }  # vqa2/val/mscoco_val2014_annotations.json

        # processed annotations contain answer_token and answer scores
        self.processed_dirs = {
            split: os.path.join(self.dir_splits[split], "processed")
            for split in self.filename_annotations.keys()
        }

        self.path_annotations_processed = {
            split: os.path.join(self.processed_dirs[split], "annotations.json")
            for split in self.processed_dirs
        }
        for k, d in self.processed_dirs.items():
            os.makedirs(d, exist_ok=True)

        self.path_answers = os.path.join(
            self.dir_dataset, f"aid_to_ans-{self.min_ans_occ}.json"
        )

        self.download()
        self.process_annotations()

        if self.features is not None:
            self.load_features()

        self.load()  # load questions and annotations

        if self.has_annotations:
            # This dictionnary will be used for evaluation
            self.qid_to_annot = {a["question_id"]: a for a in self.annotations}

        # aid_to_ans
        self.ans_to_aid = {ans: i for i, ans in enumerate(self.answers)}

    def load_questions(self, split):
        with open(self.path_questions[split]) as f:
            return json.load(f)["questions"]

    def load_original_annotations(self, split):
        with open(self.path_original_annotations[split]) as f:
            return json.load(f)["annotations"]

    def load_processed_annotations(self, split):
        with open(self.path_annotations_processed[split]) as f:
            return json.load(f)

    def load_features(self):
        if self.split == "test":
            self.feats = get_features(
                self.features, split="test2015", dir_data=self.dir_features,
            )
        else:
            self.feats = get_features(
                self.features, split="trainval2014", dir_data=self.dir_features
            )

    def get_all_tokens(self):
        tokenizer = get_tokenizer("basic_english")
        return list(
            set((token for q in self.questions for token in tokenizer(q["question"])))
        )

    def get_word_embeddings(self, name, freeze=True):
        """
        This will create the word embedding adapted to the dataset, and will cache it in the
        dataset directory for faster usage.
        """
        os.makedirs(os.path.join(self.dir_dataset, "wordembeddings"), exist_ok=True)
        path = os.path.join(self.dir_dataset, "wordembeddings", f"{name}.pth")
        tokens = self.get_all_tokens()
        if os.path.exists(path):
            w_emb = WordEmbedding(tokens, dim=get_dim_from_name(name), freeze=freeze)
            state_dict = torch.load(path)
            w_emb.load_state_dict(state_dict)
            print("done")
        else:
            print(
                f"Loading and caching {name} word embeddings for {self.name} dataset."
            )
            w_emb = WordEmbedding.from_pretrained(
                "glove.6B.300d", tokens=self.get_all_tokens(), dir_data=self.dir_data
            )
            torch.save(w_emb.state_dict(), path)
        return w_emb

    def process_annotations(self):
        """Process answers to create answer tokens,
        and precompute VQA score for faster evaluation.
        This follows the official VQA evaluation tool.
        """
        path_train = self.path_annotations_processed["train"]
        path_val = self.path_annotations_processed["val"]
        if not os.path.exists(path_train) or not os.path.exists(path_val):
            annotations_train = self.load_original_annotations("train")
            annotations_val = self.load_original_annotations("val")
            all_annotations = annotations_train + annotations_val

            print("Processing annotations")
            processor = EvalAIAnswerProcessor()

            print("\tPre-Processing answer punctuation")
            for annot in tqdm(all_annotations):

                annot["multiple_choice_answer"] = processor(
                    annot["multiple_choice_answer"]
                )
                # vqa_utils.processPunctuation(
                #     annot["multiple_choice_answer"]
                # )
                for ansDic in annot["answers"]:
                    ansDic["answer"] = processor(ansDic["answer"])
            qid_to_scores = dict()
            print("\tPre-Computing answer scores")
            for annot in tqdm(all_annotations):
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
                qid_to_scores[annot["question_id"]] = annot["scores"]
            print(f"Saving processed annotations at {path_train} and {path_val}")

            with open(self.path_annotations_processed["train"], "w") as f:
                json.dump(annotations_train, f)
            with open(self.path_annotations_processed["val"], "w") as f:
                json.dump(annotations_val, f)
            with open(os.path.join(self.dir_dataset, "qid_to_scores.json"), "w") as f:
                json.dump(qid_to_scores, f)

        #####################################
        # Processing min occurences of answer
        #####################################
        if not os.path.exists(self.path_answers):
            print(f"Removing uncommon answers")
            annotations_train = self.load_processed_annotations("train")
            annotations_val = self.load_processed_annotations("val")
            all_annotations = annotations_train + annotations_val

            occ = Counter(annot["multiple_choice_answer"] for annot in all_annotations)
            self.answers = [ans for ans in occ if occ[ans] >= self.min_ans_occ]
            print(
                f"Num answers after keeping occ >= {self.min_ans_occ}: {len(self.answers)}."
            )
            print(f"Saving answers at {self.path_answers}")
            with open(self.path_answers, "w") as f:
                json.dump(self.answers, f)

    def load(self):
        print("Loading questions")
        with open(self.path_questions[self.split]) as f:
            self.questions = json.load(f)["questions"]

        print("Loading annotations")
        if self.has_annotations:
            with open(self.path_annotations_processed[self.split]) as f:
                self.annotations = json.load(f)

        print(f"Loading aid_to_ans")
        with open(self.path_answers) as f:
            self.answers = json.load(f)

    def download(self):
        # download all splits
        for split in self.url_questions.keys():
            url_questions = self.url_questions[split]
            directory = self.dir_splits[split]
            path_questions = self.path_questions[split]
            if not os.path.exists(path_questions):
                print(f"Downloading questions at {url_questions} to {directory}")
                download_and_unzip(url_questions, directory=directory)

        for split in self.url_annotations.keys():
            url_annotations = self.url_annotations[split]
            directory = self.dir_splits[split]
            path_annotations = self.path_original_annotations[split]
            if not os.path.exists(path_annotations):
                print(f"Downloading annotations {url_annotations} to {directory}")
                download_and_unzip(url_annotations, directory=directory)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        """
        Returns a dictionnary with the following keys : {
            'image_id', 
            'question_id',
            'question', 
            'answer_type',
            'multiple_choice_answer',
            'answers',
            'image_id',
            'question_type',
            'question_id',
            'scores',
            'label'   # ground truth label to be used for the loss
        }
        Aditionnaly, if visual features are used, keys from the features will be added.
        """
        item = {"index": index}
        item.update(self.questions[index])
        if self.has_annotations:
            item.update(self.annotations[index])
            if self.label == "multilabel":
                label = torch.zeros(len(self.answers))
                for ans, score in self.annotations[index]["scores"].items():
                    if ans in self.ans_to_aid:
                        aid = self.ans_to_aid[ans]
                        label[aid] = score
                item["label"] = label
            elif self.label == "best":
                scores = self.annotations[index]["scores"]
                best_ans = max(scores, key=scores.get)
                ans_id = self.ans_to_aid[best_ans]
                item["label"] = torch.tensor(ans_id)

        if self.features is not None:
            image_id = item["image_id"]
            item.update(self.feats[image_id])

        if self.tokenize_questions:
            item["question_tokens"] = self.tokenizer(item["question"])
        return item

    @staticmethod
    def collate_fn(batch):
        no_collate_keys = ["scores", "question_id", "question_tokens"]
        result_batch = {}
        for key in batch[0]:
            if key not in no_collate_keys:
                result_batch[key] = default_collate([item[key] for item in batch])
            else:
                result_batch[key] = [item[key] for item in batch]
        return result_batch

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
        return {key: mean(score_list) if len(score_list) else 0.0 for key, score_list in scores.items()}


class VQADataModule(pl.LightningDataModule):

    dataset = VQA

    def __init__(
        self,
        dir_data: str,
        min_ans_occ=8,
        features=None,
        tokenize_question=False,
        label="multilabel",
        batch_size=512,
        num_workers=4,
    ):
        super().__init__()
        self.dir_data = dir_data
        self.label = label
        self.features = features
        self.min_ans_occ = min_ans_occ
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenize_question = tokenize_question

    def prepare_data(self):
        self.train_dataset = self.dataset(
            dir_data=self.dir_data,
            split="train",
            features=self.features,
            min_ans_occ=self.min_ans_occ,
            label=self.label,
        )

    def setup(self):
        self.val_dataset = self.dataset(
            dir_data=self.dir_data,
            split="val",
            features=self.features,
            min_ans_occ=self.min_ans_occ,
            label=self.label,
            tokenize_questions=self.tokenize_question,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = self.dataset(
            dir_data=self.dir_data,
            split="test",
            features=self.features,
            min_ans_occ=self.min_ans_occ,
            label=self.label,
            tokenize_questions=self.tokenize_question,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )


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

    filename_questions = {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "v2_OpenEnded_mscoco_val2014_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
        "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json",
    }

    filename_annotations = {
        "train": "v2_mscoco_train2014_annotations.json",
        "val": "v2_mscoco_val2014_annotations.json",
    }


class VQA2DataModule(VQADataModule):
    dataset = VQA2


class VQACP(VQA):

    DOWNLOAD_SPLITS = ["train", "test"]

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
            self.features, split="trainval", dir_data=self.dir_features,
        )

    def load_original_annotations(self):
        print("Loading annotations")
        with open(self.path_original_annotations) as f:
            self.annotations = json.load(f)

    def load(self):
        print("Loading questions")
        with open(self.path_questions) as f:
            self.questions = json.load(f)

        print("Loading processed annotations")
        with open(self.path_annotations_processed) as f:
            self.annotations = json.load(f)


class VQACPDataModule(VQADataModule):
    dataset = VQACP

    def test_dataloader(self):
        return None


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


class VQACP2DataModule(VQADataModule):
    dataset = VQACP2

    def test_dataloader(self):
        return None
