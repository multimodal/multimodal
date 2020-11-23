import multimodal.datasets
import multimodal.text
import os
from appdirs import user_data_dir
import multimodal.utils as utils

if os.environ["MULTIMODAL_DATA_DIR"]:
    DEFAULT_DATA_DIR = os.environ["MULTIMODAL_DATA_DIR"]

else:
    DEFAULT_DATA_DIR = user_data_dir("multimodal")
