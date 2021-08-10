# stdlib
import itertools
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
from torch.utils.data.dataloader import default_collate
from torchtext.data.utils import get_tokenizer

# own
from multimodal.features import get_features
from multimodal.datasets import vqa_utils
from multimodal import DEFAULT_DATA_DIR
from multimodal.utils import download_and_unzip, download_file
from multimodal.datasets.vqa_utils import EvalAIAnswerProcessor


class AbstractVQA(Dataset):
    def get_all_tokens(self) -> List:
        raise NotImplementedError()

    def evaluate(self, predictions) -> float:
        raise NotImplementedError()


class VQA(AbstractVQA):
    """
    Pytorch Dataset implementation for the VQA v1 dataset (visual question answering). 
    See https://visualqa.org/ for more details about it.
    
    When this class is instanciated, data will be downloaded in the directory specified by the ``dir_data`` parameter.
    Pre-processing of questions and answers will take several minutes.

    When the ``features`` argument is specified, visual features will be downloaded as well. About 60Go will be 
    necessary for downloading and extracting features.

    Args:
        dir_data (str): dir for the multimodal cache (data will be downloaded in a vqa2/ folder inside this directory
        features (str|object): which visual features should be used. Choices: ``coco-bottomup`` or ``coco-bottomup-36``
            You can also give directly the feature instance.
        split (str): Which t [``train``, ``val``, ``test``]
        dir_features (str): directory to download features. If None, defaults to $dir_data/features
        label (str): either `multilabel`, or `best`. For `multilabel`, GT scores for questions are
            given by the score they are assigned by the VQA evaluation. 
            If `best`, GT is the label of the top answer.
        tokenize_questions (bool): If True, preprocessing will tokenize questions into tokens.
            The tokens are stored in item["question_tokens"].
        load (bool): default `True`. If false, then the questions annotations and questions will not be loaded
            in memory. This is useful if you want only to download and process the data.
    """

    SPLITS = ["train", "val", "test", "test-dev"]

    UNZIP = True

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
        min_ans_occ=9,
        dir_features=None,
        label="multilabel",
        tokenize_questions=False,
        load=True,
    ):
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

        self.dir_splits["test-dev"] = self.dir_splits["test"]  # same directory

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

        self._download()
        self._process_annotations()

        if self.features is not None and type(features) == str:
            self._load_features()
        elif self.features is not None and isinstance(self.features, type):
            # object is given, do nothing.
            pass

        if load:
            self._load()  # load questions and annotations

            if self.has_annotations:
                # This dictionnary will be used for evaluation
                self.qid_to_annot = {a["question_id"]: a for a in self.annotations}

            # aid_to_ans
            self.ans_to_aid = {ans: i for i, ans in enumerate(self.answers)}

    @classmethod
    def download_and_process(cls, dir_data):
        cls(dir_data=dir_data, split="train", load="False")

    def _load_questions(self, split):
        with open(self.path_questions[split]) as f:
            return json.load(f)["questions"]

    def _load_original_annotations(self, split):
        with open(self.path_original_annotations[split]) as f:
            return json.load(f)["annotations"]

    def _load_processed_annotations(self, split):
        with open(self.path_annotations_processed[split]) as f:
            return json.load(f)

    def _load_features(self):
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

    def get_all_questions(self):
        return (q["question"] for q in self.questions)

    def _process_annotations(self):
        """Process answers to create answer tokens,
        and precompute VQA score for faster evaluation.
        This follows the official VQA evaluation tool.
        """
        paths = [self.path_annotations_processed[split] for split in self.url_annotations]
        # path_train = self.path_annotations_processed["train"]
        # path_val = self.path_annotations_processed["val"]
        if any(not os.path.exists(p) for p in paths):
            annotations = [self._load_original_annotations(split) for split in self.url_annotations]
            all_annotations = list(itertools.chain(*annotations))
            # annotations_train = self._load_original_annotations("train")
            # annotations_val = self._load_original_annotations("val")
            # all_annotations = annotations_train + annotations_val

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

            for i, split in enumerate(self.url_annotations):
                print(f"Saving processed annotations for split {split} at path {self.path_annotations_processed[split]}")
                with open(self.path_annotations_processed[split], "w") as f:
                    json.dump(annotations[i], f)

            with open(os.path.join(self.dir_dataset, "qid_to_scores.json"), "w") as f:
                json.dump(qid_to_scores, f)

        #####################################
        # Processing min occurences of answer
        #####################################
        if not os.path.exists(self.path_answers):
            print(f"Removing uncommon answers")
            annotations = [self._load_processed_annotations(split) for split in self.url_annotations]
            all_annotations = itertools.chain(*annotations)
            occ = Counter(annot["multiple_choice_answer"] for annot in all_annotations)
            self.answers = [ans for ans in occ if occ[ans] >= self.min_ans_occ]
            print(
                f"Num answers after keeping occ >= {self.min_ans_occ}: {len(self.answers)}."
            )
            print(f"Saving answers at {self.path_answers}")
            with open(self.path_answers, "w") as f:
                json.dump(self.answers, f)

    def _load(self):
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

    def _download(self):
        # download all splits
        for split in self.url_questions.keys():
            url_questions = self.url_questions[split]
            directory = self.dir_splits[split]
            path_questions = self.path_questions[split]
            if not os.path.exists(path_questions):
                print(f"Downloading questions at {url_questions} to {directory}")
                if self.UNZIP:
                    download_and_unzip(url_questions, directory=directory)
                else:
                    download_file(url_questions, directory=directory)

        for split in self.url_annotations.keys():
            url_annotations = self.url_annotations[split]
            directory = self.dir_splits[split]
            path_annotations = self.path_original_annotations[split]
            if not os.path.exists(path_annotations):
                print(f"Downloading annotations {url_annotations} to {directory}")
                if self.UNZIP:
                    download_and_unzip(url_annotations, directory=directory)
                else:
                    download_file(url_annotations, directory=directory)

    def __len__(self):
        """
        Returns the number of (question-image-answer) items in the dataset.
        """
        return len(self.questions)

    def __getitem__(self, index):
        """
        Returns a dictionnary with the following keys 

        .. code-block::

            {
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
        """
        Use this method to collate batches of data.
        """
        no_collate_keys = ["scores", "question_id", "question"]
        result_batch = {}
        for key in batch[0]:
            if key not in no_collate_keys:
                result_batch[key] = default_collate([item[key] for item in batch])
            else:
                result_batch[key] = [item[key] for item in batch]
        return result_batch

    def evaluate(self, predictions):
        """
        Evaluates a list of predictions, according to the VQA evaluation protocol. See https://visualqa.org/evaluation.html.

        Args:
            predictions (list): List of dictionnaries containing ``question_id`` and ``answer`` keys. The answer must be specified 
                as a string.

        Returns:
            A dict of floats containing scores for "overall", "yes/no", number", and "other" questions.
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
        return {
            key: mean(score_list) if len(score_list) else 0.0
            for key, score_list in scores.items()
        }


class VQA2(VQA):
    """
    Pytorch Dataset implementation for the VQA v2 dataset (visual question answering). 
    See https://visualqa.org/ for more details about it.
    
    When this class is instanciated, data will be downloaded in the directory specified by the ``dir_data`` parameter.
    Pre-processing of questions and answers will take several minutes.

    When the ``features`` argument is specified, visual features will be downloaded as well. About 60Go will be 
    necessary for downloading and extracting features.

    Args:
        dir_data (str): dir for the multimodal cache (data will be downloaded in a vqa2/ folder inside this directory
        features (str): which visual features should be used. Choices: ``coco-bottomup`` or ``coco-bottomup-36``
        split (str): Which t [``train``, ``val``, ``test``]
        dir_features (str): directory to download features. If None, defaults to $dir_data/features
        label (str): either `multilabel`, or `best`. For `multilabel`, GT scores for questions are
            given by the score they are assigned by the VQA evaluation. 
            If `best`, GT is the label of the top answer.
        tokenize_questions (bool): If True, preprocessing will tokenize questions into tokens.
            The tokens are stored in item["question_tokens"].
    """

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


class VQACP(VQA):
    """  Pytorch Dataset implementation for the VQA-CP v1 dataset (visual question answering). 
    See https://www.cc.gatech.edu/grads/a/aagrawal307/vqa-cp/ for more details about it.
    
    When this class is instanciated, data will be downloaded in the directory specified by the ``dir_data`` parameter.
    Pre-processing of questions and answers will take several minutes.

    When the ``features`` argument is specified, visual features will be downloaded as well. About 60Go will be 
    necessary for downloading and extracting features.

    Args:
        dir_data (str): dir for the multimodal cache (data will be downloaded in a vqa2/ folder inside this directory
        features (str): which visual features should be used. Choices: ``coco-bottomup`` or ``coco-bottomup-36``
        split (str): Which t [``train``, ``val``, ``test``]
        dir_features (str): directory to download features. If None, defaults to $dir_data/features
        label (str): either `multilabel`, or `best`. For `multilabel`, GT scores for questions are
            given by the score they are assigned by the VQA evaluation. 
            If `best`, GT is the label of the top answer.
        tokenize_questions (bool): If True, preprocessing will tokenize questions into tokens.
            The tokens are stored in item["question_tokens"].
    """

    DOWNLOAD_SPLITS = ["train", "test"]
    UNZIP = False

    name = "vqacp"

    url_questions = {
        "train": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_train_questions.json",
        "test": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_test_questions.json",
    }

    url_annotations = {
        "train": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_train_annotations.json",
        "test": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_test_annotations.json",
    }

    filename_questions = {
        "train": "vqacp_v1_train_questions.json",
        "test": "vqacp_v1_test_questions.json",
    }

    filename_annotations = {
        "train": "vqacp_v1_train_annotations.json",
        "test": "vqacp_v1_test_annotations.json",
    }


    def _load_questions(self, split):
        with open(self.path_questions[split]) as f:
            return json.load(f)

    def _load_original_annotations(self, split):
        with open(self.path_original_annotations[split]) as f:
            return json.load(f)


    def _load(self):
        print("Loading questions")
        with open(self.path_questions[self.split]) as f:
            self.questions = json.load(f)

        print("Loading annotations")
        if self.has_annotations:
            with open(self.path_annotations_processed[self.split]) as f:
                self.annotations = json.load(f)

        print(f"Loading aid_to_ans")
        with open(self.path_answers) as f:
            self.answers = json.load(f)

class VQACP2(VQACP):
    """  Pytorch Dataset implementation for the VQA-CP v2 dataset (visual question answering). 
    See https://www.cc.gatech.edu/grads/a/aagrawal307/vqa-cp/ for more details about it.
    
    When this class is instanciated, data will be downloaded in the directory specified by the ``dir_data`` parameter.
    Pre-processing of questions and answers will take several minutes.

    When the ``features`` argument is specified, visual features will be downloaded as well. About 60Go will be 
    necessary for downloading and extracting features.

    Args:
        dir_data (str): dir for the multimodal cache (data will be downloaded in a vqa2/ folder inside this directory
        features (str): which visual features should be used. Choices: ``coco-bottomup`` or ``coco-bottomup-36``
        split (str): Which t [``train``, ``val``, ``test``]
        dir_features (str): directory to download features. If None, defaults to $dir_data/features
        label (str): either `multilabel`, or `best`. For `multilabel`, GT scores for questions are
            given by the score they are assigned by the VQA evaluation. 
            If `best`, GT is the label of the top answer.
        tokenize_questions (bool): If True, preprocessing will tokenize questions into tokens.
            The tokens are stored in item["question_tokens"].
    """

    name = "vqacp2"

    url_questions = {
        "train": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json",
        "test": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json",
    }

    url_annotations = {
        "train": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json",
        "test": "https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json",
    }

    filename_questions = {
        "train": "vqacp_v2_train_questions.json",
        "test": "vqacp_v2_test_questions.json",
    }

    filename_annotations = {
        "train": "vqacp_v2_train_annotations.json",
        "test": "vqacp_v2_test_annotations.json",
    }
