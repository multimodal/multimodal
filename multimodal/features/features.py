# from multimodal.features.bottomup import 
import os
import numpy as np

from multimodal.features import COCOBottomUpFeatures


def get_features(name, split, dir_data, args={}):
    if dir_data is None:
        dir_data = os.environ.get("MULTIMODAL_FEATURES_DIR")
    if dir_data is None:
        raise ValueError(
            f"No value for dir_cache specified. Set "
            "variable dir_cache, or "
            "environment variable MULTIMODAL_FEATURES_DIR"
        )
    if name == "coco-bottomup":
        features = COCOBottomUpFeatures(dir_data=dir_data, features=split)
    elif name == "coco-bottomup-36":
        features = COCOBottomUpFeatures(dir_data=dir_data, features=split + "_36")
    # elif name == "coco-bottomup-36-sqlite":
    #     features = COCOBottomUpFeaturesSqlite(dir_data=dir_data, features=split + "_36")
    # elif name == "coco-bottomup-36-tables":
    #     features = COCOBottomUpFeaturesPyTables(dir_data=dir_data, features=split + "_36")
    # elif name == "coco-bottomup-36-tables2":
    #     features = COCOBottomUpFeaturesPyTables2(dir_data=dir_data, features=split + "_36")
    # elif name == "coco-bottomup-tables2":
    #     features = COCOBottomUpFeaturesPyTables2(dir_data=dir_data, features=split, **args)
    elif name == "mock-features":
        features = MockFeatures()
    else:
        raise ValueError(f"No features named {name}")
    return features

class MockFeatures:

    def __getitem__(self, index):
        return {
             "image_id": 0 ,
              "image_w": 128,
              "image_h": 128,
              "num_boxes": 36,
              "boxes": np.zeros((36, 4)),
              "features": np.zeros((36, 2048)),
        }
