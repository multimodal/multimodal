import os

# import urllib
from appdirs import user_data_dir
from torch.utils.data import Dataset
import zipfile
import json
from pySmartDL import SmartDL
from collections import defaultdict
from tqdm import tqdm

from multimodal.features import get_features

def download(url, directory):
    basename = get_basename(url)
    path = os.path.join(directory, basename)
    os.makedirs(directory, exist_ok=True)
    obj = SmartDL(url, path)
    obj.start()
    return obj.get_dest()


def get_basename(url):
    return url.split("/")[-1]


def download_and_unzip(url, directory):
    basename = get_basename(url)
    path = os.path.join(directory, basename)
    os.makedirs(directory, exist_ok=True)
    SmartDL(url, path).start()
    # urllib.request.urlretrieve(url, path)
    with zipfile.ZipFile(path) as f:
        f.extractall(path=directory)


class COCO(Dataset):

    annotation_urls = {
        "trainval2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "trainval2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "test2014": "http://images.cocodataset.org/annotations/image_info_test2014.zip",
        "test2015": "http://images.cocodataset.org/annotations/image_info_test2015.zip",
        "test2017": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
    }

    splits_to_download_split = {
        "train2014": "trainval2014",
        "train2017": "trainval2017",
        "val2014": "trainval2014",
        "val2017": "trainval2017",
        "test2014": "test2014",
        "test2015": "test2015",
        "test2017": "test2017",
    }

    def __init__(
        self,
        cache_dir=None,
        split="train2014",
        task="captions",
        dir_images=None,
        features=None,
        dir_features=None,
        itemize="image",
    ):
        """
        cache_dir: where the annotations will be downloaded  (a folder named COCO will be created)
        split: either [train2014, val2014, train2017, val2017, test2014, test2015, test2017]
        task: [captions, instances, person_keypoints, image_info]
        itemize: one of [image, task]
        """
        # download zipfile
        if cache_dir is None:
            cache_dir = user_data_dir(appname="multimodal")

        self.task = task
        self.cache_dir = os.path.join(cache_dir, "coco")
        self.split = split
        self.itemize = itemize
        self.download_split = self.splits_to_download_split[self.split]

        task_file_path = self.get_task_file_path()
        if not os.path.exists(task_file_path):
            self.download()

        # load task
        with open(task_file_path) as f:
            self.data = json.load(f)

        if self.itemize == "image":
            # gather data for every image
            self.data_for_image = defaultdict(list)
            for item in tqdm(self.data["annotations"]):
                image_id = item["image_id"]
                self.data_for_image[image_id].append(item)
        elif self.itemize == "task":
            self.image_id_to_image = {}
            for image in tqdm(self.data["images"]):
                self.image_id_to_image[image["id"]] = image

    def load_features(self):
        if "test" in self.split:
            self.feats = get_features(
                self.features, split=self.split, dir_cache=self.dir_features,
            )
        elif self.split == "trainval2014":  # default split for bottom-up
            self.feats = get_features(
                self.features, split="trainval", dir_cache=self.dir_features
            )
        elif self.Split == "trainval2017":
            raise NotImplementedError()

    def get_task_file_path(self):
        task_file = f"annotations/{self.task}_{self.split}.json"
        return os.path.join(self.cache_dir, self.download_split, task_file)

    def download(self):
        directory = os.path.join(self.cache_dir, self.download_split)
        url = self.annotation_urls[self.splits_to_download_split[self.split]]
        print(f"Downloading file from {url} to {directory}.")
        download_and_unzip(url, directory)

    def __len__(self):
        if self.itemize == "image":
            return len(self.data["images"])
        elif self.itemize == "task":
            return len(self.data["annotations"])

    def __getitem__(self, index):
        if self.itemize == "image":
            image_info = self.data["images"][index]
            image_id = image_info["id"]
            return {
                "image": image_info,
                "annotations": self.data_for_image[image_id],
            }
        elif self.itemize == "task":
            data = self.data["annotations"][index]
            image_id = data["image_id"]
            image_info = self.image_id_to_image[image_id]
            return {
                "image": image_info,
                "annotations": data,
            }
