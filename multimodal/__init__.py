from appdirs import user_data_dir
import os

if os.environ.get("MULTIMODAL_DATA_DIR", None):
    DEFAULT_DATA_DIR = os.environ["MULTIMODAL_DATA_DIR"]
else:
    DEFAULT_DATA_DIR = user_data_dir("multimodal")

import multimodal.datasets
import multimodal.text
import multimodal.utils as utils
