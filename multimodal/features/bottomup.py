"""
Vision features for muldimodal tasks like Image Captioning, VQA or image retrieval
"""
# std
import os
import zipfile
import csv
import base64
import pickle
import sys

# packages
from tqdm import tqdm
import shutil
import numpy as np

# multimodal
from multimodal.utils import download_file

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]


def get_basename(url):
    return url.split("/")[-1]



class COCOBottomUpFeatures:
    """
    Bottom up features for the COCO dataste
    """

    name = "coco-bottom-up"

    urls = {
        "trainval2014_36": "https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip",  # trainval2014
        "test2015_36": "https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip",
        "test2014_36": "https://imagecaption.blob.core.windows.net/imagecaption/test2014_36.zip",
        "trainval2014": "https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip",  # trainval2014
        "test2015": "https://imagecaption.blob.core.windows.net/imagecaption/test2015.zip",
        "test2014": "https://imagecaption.blob.core.windows.net/imagecaption/test2014.zip",
    }

    tsv_paths = {
        "trainval2014_36": "trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv",
        "test2015_36": "test2015_36/test2014_resnet101_faster_rcnn_genome_36.tsv",
        "test2014_36": "test2014_36/test2014_resnet101_faster_rcnn_genome_36.tsv",
        "trainval2014": "trainval/trainval_resnet101_faster_rcnn_genome.tsv",
        "test2015": "test2015/test2014_resnet101_faster_rcnn_genome.tsv",
        "test2014": "test2014/test2014_resnet101_faster_rcnn_genome.tsv",
    }

    def __init__(self, features="test2014_36", dir_data=None):
        self.features_name = features
        self.featsfile = None  # Lazy loading of zipfile
        self.dir_data = os.path.join(dir_data, "features", self.name)
        os.makedirs(self.dir_data, exist_ok=True)
        self.featspath = os.path.join(self.dir_data, features + ".zipfeat")

        # processing
        if not os.path.exists(self.featspath):
            url = self.urls[features]
            path_download = download_file(url, self.dir_data)
            print("Processing file")
            self.process_file(path_download, self.featspath)

    def __getitem__(self, image_id: str):
        self.check_open()
        return pickle.loads(self.featsfile.read(str(image_id)))

    def check_open(self):
        if self.featsfile is None:
            self.featsfile = zipfile.ZipFile(self.featspath)

    def keys(self):
        self.check_open()
        return self.featsfile.namelist()

    def process_file(self, path_infile, outfile):
        directory = os.path.dirname(path_infile)
        tsv_path = self.tsv_paths[self.features_name]
        try:
            if not os.path.exists(tsv_path):
                print(f"Unzipping file at {path_infile}")
                with zipfile.ZipFile(path_infile, "r") as zip_ref:
                    zip_ref.extractall(directory)
            names = set()
            num_duplicates = 0
            print(f"Processing file {tsv_path}")
        except Exception:
            os.remove(os.path.join(self.dir_data, self.features_name))
            raise
        try:
            outzip = zipfile.ZipFile(outfile, "w")
            with open(tsv_path, "r") as tsv_in_file:
                reader = csv.DictReader(
                    tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES
                )
                for item in tqdm(reader):
                    item["image_id"] = int(item["image_id"])
                    item["image_h"] = int(item["image_h"])
                    item["image_w"] = int(item["image_w"])
                    item["num_boxes"] = int(item["num_boxes"])
                    if item["image_id"] in names:
                        print(f"Duplicate {item['image_id']}")
                        num_duplicates += 1
                        continue
                    for field in ["boxes", "features"]:
                        item[field] = np.frombuffer(
                            base64.decodestring(item[field].encode("ascii")),
                            dtype=np.float32,
                        ).reshape((item["num_boxes"], -1))
                    names.add(item["image_id"])
                    with outzip.open(str(item["image_id"]), "w") as itemfile:
                        pickle.dump(item, itemfile)
            print(f"Num duplicates : {num_duplicates}")
            outzip.close()
        except Exception:
            outzip.close()
            os.remove(outfile)
            raise
        # remove tsv
        print("Deleting tsv from disk")
        shutil.rmtree(os.path.join(self.dir_data, self.features_name))
