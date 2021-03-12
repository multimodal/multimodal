from torch.utils.data.dataloader import DataLoader
from multimodal.datasets import VQA, VQA2, VQACP, VQACP2

try:
    import pytorch_lightning as pl
except:
    raise ImportError(
        "You need to install pytorch_lightning in order to use multimodal.datasets.lightning"
    )


class VQADataModule(pl.LightningDataModule):

    dataset = VQA

    def __init__(
        self,
        dir_data: str,
        min_ans_occ=8,
        features="coco-bottomup-36",
        tokenize_question=False,
        label="multilabel",
        batch_size=512,
        num_workers=4,
    ):
        super().__init__()
        self.dir_data = dir_data
        self.label = label
        self.features = features
        self.min_ans_occ = min_ans_occ
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenize_question = tokenize_question

    def prepare_data(self):
        dset = self.dataset(
            dir_data=self.dir_data,
            split="train",
            features=self.features,
            min_ans_occ=self.min_ans_occ,
            label=self.label,
            load=False,
        )
        self.num_ans = len(dset.answers)

    def setup(self):
        self.train_dataset = self.dataset(
            dir_data=self.dir_data,
            split="train",
            features=self.features,
            min_ans_occ=self.min_ans_occ,
            label=self.label,
        )

        self.val_dataset = self.dataset(
            dir_data=self.dir_data,
            split="val",
            features=self.features,
            min_ans_occ=self.min_ans_occ,
            label=self.label,
            tokenize_questions=self.tokenize_question,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = self.dataset(
            dir_data=self.dir_data,
            split="test",
            features=self.features,
            min_ans_occ=self.min_ans_occ,
            label=self.label,
            tokenize_questions=self.tokenize_question,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class VQA2DataModule(VQADataModule):
    dataset = VQA2


class VQACPDataModule(VQADataModule):
    dataset = VQACP

    def test_dataloader(self):
        return None


class VQACP2DataModule(VQADataModule):
    dataset = VQACP2

    def test_dataloader(self):
        return None
