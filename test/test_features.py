from multimodal.features.bottomup import COCOBottomUpFeatures
import tempfile
import shutil
import os

def test_bottomup():
    
    with tempfile.TemporaryDirectory() as d:
        path_features = os.path.join(d, "features", "coco-bottom-up")
        os.makedirs(path_features)
        shutil.copy("test/data/trainval_36.zip", path_features)
        features = COCOBottomUpFeatures(features="trainval2014_36", dir_data=d)
        list_images = features.keys()
        assert len(list_images) == 10
        feats = features[list_images[0]]
        assert isinstance(feats, dict)
        assert feats['image_id'] == 150367
        assert feats['image_w'] == 640
        assert feats['image_h'] == 480
        assert feats['num_boxes'] == 36
        assert feats['boxes'].shape == (36, 4)
        assert feats['features'].shape == (36, 2048)
