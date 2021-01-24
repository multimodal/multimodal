Bottom-Up Top-Down Object Features
==================================

Those visual features were introduced by Anderson et. al. in the paper 
`Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering <https://arxiv.org/abs/1707.07998>`_.

They are extracted with a Faster R-CNN, and trained on the Visual Genome dataset to detect objects and
their attributes (shapes, colors...).

**multimodal** provides a class to download and use those features extracted on the COCO image dataset.
They can be used for most Visual Question Answering and Captionning that use the this dataset for images.


.. code-block:: python

    from multimodal.features import COCOBottomUpFeatures

    bottomup = COCOBottomUpFeatures(
        features="coco-bottomup-36",
        dir_data="/data/multimodal",
    )
    image_id = 13455
    feats = bottomup[image_id]
    print(feats.keys())
    # ['image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    print(feats["features"].shape)  # numpy array
    # (36, 2048)


.. autoclass:: multimodal.features.COCOBottomUpFeatures
    :members:
    :private-members:
    :special-members: __getitem__
