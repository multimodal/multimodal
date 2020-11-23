from multimodal.features import COCOBottomUpFeatures
import os


def get_features(name, split, dir_cache):
    if dir_cache is None:
        dir_cache = os.environ.get("MULTIMODAL_FEATURES_DIR")
    if dir_cache is None:
        raise ValueError(
            f"No value for dir_cache specified. Set "
            "variable dir_cache, or "
            "environment variable MULTIMODAL_FEATURES_DIR"
        )
    if name == "coco-bottom-up":
        features = COCOBottomUpFeatures(dir_cache=dir_cache, features=split)
    elif name == "coco-bottom-up-36":
        features = COCOBottomUpFeatures(dir_cache=dir_cache, features=split + "_36")
    else:
        raise ValueError(f"No features named {name}")
    return features
