from multimodal.features import COCOBottomUpFeatures


def get_features(name, split, dir_cache):
    if name == "coco-bottom-up":
        features = COCOBottomUpFeatures(dir_cache=dir_cache, features=split)
    elif name == "coco-bottom-up-36":
        features = COCOBottomUpFeatures(dir_cache=dir_cache, features=split + "_36")
    return features
