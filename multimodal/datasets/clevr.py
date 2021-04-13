from genericpath import exists
import os
import json

from multimodal.utils import Task, download_and_unzip
from multimodal import DEFAULT_DATA_DIR

from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch


class CLEVR(Dataset):
    """
    CLEVR: A Diagnostic Dataset for
    Compositional Language and Elementary Visual Reasoning.
    
    See https://cs.stanford.edu/people/jcjohns/clevr/

    Warning: instanciating this class will download a 18Gb file to the multimodal data directory
    (by default in your applications data). You can specify 
    the multimodal data directory by specifying the ``dir_data`` argument, or specifying it in your path.

    Args:
        dir_data (str): dir for the multimodal cache (data will be downloaded in a clevr/ folder inside this directory
        split (str): either train, val or test
        transform: torchvision transform applied to images. By default, only ToTensor.
    """

    url = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"

    def __init__(
        self, dir_data=None, split="train", transform=torchvision.transforms.ToTensor()
    ):
        super().__init__()
        if dir_data is None:
            dir_data = DEFAULT_DATA_DIR
        self.split = split
        self.dir_dataset = os.path.join(dir_data, "datasets", "clevr")
        self.transform = transform
        os.makedirs(self.dir_dataset, exist_ok=True)
        self.download_and_process(self.dir_dataset)
        # open data
        with open(self._get_path_questions(self.dir_dataset, self.split)) as f:
            self.questions = json.load(f)["questions"]

        with open(os.path.join(self.dir_dataset, "answers.json")) as f:
            self.aid_to_ans = json.load(f)
        self.ans_to_aid = {ans: id for (id, ans) in enumerate(self.aid_to_ans)}

    @classmethod
    def _get_path_questions(cls, dir_dataset, split):
        return os.path.join(
            dir_dataset, f"CLEVR_v1.0/questions/CLEVR_{split}_questions.json"
        )

    @classmethod
    def download_and_process(cls, dir_dataset):
        task = Task(dir_dataset, "download")
        if not task.is_done():
            print("Downloading CLEVR")
            download_and_unzip(cls.url, directory=dir_dataset)
        task.mark_done()
        path_answers = os.path.join(dir_dataset, "answers.json")
        if not os.path.exists(path_answers):
            print("Processing answers")
            with open(cls._get_path_questions(dir_dataset, split="train")) as f:
                train_questions = json.load(f)["questions"]
                all_answers = list(set(q["answer"] for q in train_questions))
                with open(path_answers, "w") as f:
                    json.dump(all_answers, f)

    def __getitem__(self, index: int):
        """
        Returns a dictionnary with the following keys:

        .. code-block::

            {
                "index",
                "question",
                "answer":,
                "question_family_index":,
                "image_filename":,
                "image_index":,
                "image"
                "label",
            }

        Note that you can recover the program for an example by using the index:

        .. code-block:: python

            index = item["index"][0]  #  first item of batch
            program = clevr.questions[index]["program"]

        """
        q = self.questions[index]
        img_path = os.path.join(
            self.dir_dataset, "CLEVR_v1.0", "images", self.split, q["image_filename"]
        )
        # add image data
        target = torch.zeros(len(self.aid_to_ans))
        ans_id = self.ans_to_aid[q["answer"]]

        im = Image.open(img_path)
        if self.transform is not None:
            im = self.transform(im)
        item = {
            "index": index,
            "question": q["question"],
            "answer": q["answer"],
            "question_family_index": q["question_family_index"],
            "image_filename": q["image_filename"],
            "image_index": q["image_index"],
            "image": im,
            "label": torch.tensor(ans_id),
        }
        return item

    def __len__(self) -> int:
        return len(self.questions)
