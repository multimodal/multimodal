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
import numpy as np
from pySmartDL import SmartDL
import tables as tb

# multimodal
from multimodal import DEFAULT_DATA_DIR

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]


class Metadata(tb.IsDescription):
    image_id = tb.Int32Col()
    image_h = tb.Int32Col()
    image_w = tb.Int32Col()
    num_boxes = tb.Int32Col()
    start_position = tb.Int32Col()


class COCOBottomUpFeatures:
    """
    Bottom up features for the COCO dataset

    Args:
        features (str): one of [``trainval2014_36``, ``trainval2014``, ``test2014_36``, ``test2014``, ``test2015-36``, ``test2015``]. 
            Specifies the split, and the number of detected objects. _36 means 36 objetcs are detected in every image, and otherwise,
            the number is based on a detection threshold, between 10 and 100 objects.
        dir_data (str): Directory where multimodal data will be downloaded. You need at least 60Go for downloading
            and extracting the features.
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
        "trainval2014_36": ["trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv"],
        "test2015_36": ["test2015_36/test2014_resnet101_faster_rcnn_genome_36.tsv"],
        "test2014_36": ["test2014_36/test2014_resnet101_faster_rcnn_genome_36.tsv"],
        "trainval2014": [
            "trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0",
            "trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1",
            "trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv",
            "trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv",
        ],
        "test2015": ["test2015/test2014_resnet101_faster_rcnn_genome.tsv"],
        "test2014": ["test2014/test2014_resnet101_faster_rcnn_genome.tsv"],
    }

    def __init__(self, features: str, dir_data: str=None):
        self.features_name = features
        self.db = None  # Lazy loading of zipfile
        dir_data = dir_data or DEFAULT_DATA_DIR
        self.dir_data = os.path.join(dir_data, "features", self.name)
        os.makedirs(self.dir_data, exist_ok=True)
        self.featspath = os.path.join(self.dir_data, features + ".tables")

        # processing
        if not os.path.exists(self.featspath):
            path_download = self.download()
            print("Processing file")
            self._process_file(path_download, self.featspath)

    @classmethod
    def download_and_process(cls, name, dir_data):
        cls(feature=name, dir_data=dir_data)

    def download(self):
        url = self.urls[self.features_name]
        dl = SmartDL(url, self.dir_data)
        destination = dl.get_dest()
        if not os.path.exists(dl.get_dest()):
            dl.start()
        return destination

    def __getitem__(self, image_id: int):
        """
        Get the features. 

        Args:
            image_id (str|int): The id of the image in COCO dataset.

        Returns:
            A dictionnary containing the following keys::

                {
                    'image_id',
                    'image_h': height
                    'image_w': width
                    'num_boxes': number of objects
                    'boxes': Numpy array of shape (N, 4) containing bounding box coordinates
                    'features': Numpy array of shape (N, 2048) containing features.
                }
        """
        self._check_open()
        data = self.db.metadata.read_where(f"image_id=={image_id}")[0]
        start_position = data["start_position"]
        num_boxes = data["num_boxes"]
        data = {
            field: data[field]
            for field in ["image_id", "image_w", "image_h", "num_boxes"]
        }
        features = self.db.features[start_position : start_position + num_boxes]
        boxes = self.db.boxes[start_position : start_position + num_boxes]
        data["features"] = features
        data["boxes"] = boxes
        return data

    def _check_open(self):
        if self.db is None:
            self.db = tb.open_file(self.featspath).root

    def keys(self):
        """
        Returns:
            list: List of all keys
        """
        self._check_open()
        return list(self.db.metadata.read(field="image_id"))

    def _process_file(self, path_infile: str, outpath: str):
        directory = os.path.dirname(path_infile)
        tsv_paths = self.tsv_paths[self.features_name]
        if type(tsv_paths) == str:
            tsv_paths = [tsv_paths]

        tsv_paths = [os.path.join(directory, path) for path in tsv_paths]
        last_tsv = tsv_paths[-1]
        try:
            if not os.path.exists(last_tsv):
                print(f"Unzipping file at {path_infile}")
                with zipfile.ZipFile(path_infile, "r") as zip_ref:
                    zip_ref.extractall(directory)
            names = set()
            num_duplicates = 0
            print(f"Processing files {tsv_paths}")
        except Exception:
            os.remove(os.path.join(self.dir_data, self.features_name))
            raise
        try:
            outfile = tb.open_file(outpath, mode="w")
        except Exception:
            os.remove(outpath)
            raise
        try:
            table = outfile.create_table(
                outfile.root, "metadata", Metadata, expectedrows=123287
            )
            array_feats = outfile.create_earray(
                outfile.root,
                "features",
                shape=(0, 2048),
                atom=tb.Float32Atom(),
                expectedrows=36 * 123287,
            )
            array_boxes = outfile.create_earray(
                outfile.root,
                "boxes",
                shape=(0, 4),
                atom=tb.Float32Atom(),
                expectedrows=36 * 123287,
            )

            feat = table.row
            table.cols.image_id.create_index()

            pbar = tqdm(total=123287, desc="Converting features to PyTables")
            for tsv_p in tsv_paths:
                with open(tsv_p, "r") as tsv_in_file:
                    reader = csv.DictReader(
                        tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES
                    )
                    start_position = 0
                    for item in reader:
                        pbar.update(1)
                        num_boxes = int(item["num_boxes"])
                        feat["image_id"] = int(item["image_id"])
                        feat["image_h"] = int(item["image_h"])
                        feat["image_w"] = int(item["image_w"])
                        feat["num_boxes"] = num_boxes
                        feat["start_position"] = start_position
                        start_position += int(item["num_boxes"])
                        if item["image_id"] in names:
                            print(f"Duplicate {item['image_id']}")
                            num_duplicates += 1
                            continue
                        for field in ["boxes", "features"]:
                            item[field] = np.frombuffer(
                                base64.decodebytes(item[field].encode("ascii")),
                                dtype=np.float32,
                            ).reshape((num_boxes, -1))
                        array_boxes.append(item["boxes"])
                        array_feats.append(item["features"])
                        feat.append()
            table.flush()
            print(f"Num duplicates : {num_duplicates}")
            outfile.close()
            pbar.close()
        except Exception:
            outfile.close()
            os.remove(outpath)
            raise
        # remove tsv
        print("Deleting tsv from disk")
        # os.remove(tsv_path)
