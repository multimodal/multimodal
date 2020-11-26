from pySmartDL import SmartDL
import os
from shutil import unpack_archive

def get_basename(url):
    return url.split("/")[-1]


def download_file(url, directory, filename=None):
    os.makedirs(directory, exist_ok=True)
    path = directory
    if filename:
        path = os.path.join(directory, filename)
    print(f"Downloading file from {url} at {path}")
    obj = SmartDL(url, path)
    obj.start()
    return obj.get_dest()


def download_and_unzip(url, directory, filename=None):
    dest = download_file(url, directory, filename=filename)
    unpack_archive(dest, extract_dir=directory)
