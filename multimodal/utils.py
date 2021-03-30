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
    obj = SmartDL(url, path)
    obj.start()
    return obj.get_dest()


def download_and_unzip(url, directory=None, filename=None, path=None):
    """
    """
    if directory is None:
        if path is None:
            raise ValueError("Either path or directory must be specified")
        directory = os.path.basename(path)
    dest = download_file(url, directory, filename=filename)
    unpack_archive(dest, extract_dir=directory)


class Task:
    def __init__(self, path, name):
        self.path = os.path.join(path, name + ".done")

    def is_done(self):
        return os.path.exists(self.path)

    def mark_done(self):
        open(self.path, "a").close()

