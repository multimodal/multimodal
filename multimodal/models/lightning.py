from multimodal.datasets.vqa import AbstractVQA
import pytorch_lightning as pl
import torch
import torch.nn as nn

class VQALightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        train_dataset: AbstractVQA = None,
        val_dataset: AbstractVQA = None,
        tokenizer=None,
    ):
        super().__init__()
        self.model = model
        self.loss = nn.BCEWithLogitsLoss()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        logits = self.one_forward(batch)["logits"]
        loss = self.loss(logits, batch["label"])
        self.log("train_loss", loss)
        accuracy = self.accuracy(logits, batch, self.train_dataset)
        for key in accuracy:
            self.log(f"Accuracy/Train/{key}", accuracy[key])
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.one_forward(batch)["logits"]
        loss = self.loss(logits, batch["label"])
        self.log("val_loss", loss)
        # accuracy
        accuracy = self.accuracy(logits, batch, self.val_dataset)
        for key in accuracy:
            self.log(f"Accuracy/Val/{key}", accuracy[key])
        return loss

    def one_forward(self, batch):
        if self.tokenizer is not None:
            question_tokens = self.tokenizer(batch["question"], tensor_type="pt")
            batch["question_tokens"] = question_tokens.to(
                device=batch["features"].device
            )
            # breakpoint()
        return self.model(batch)

    def test_step(self, batch, batch_idx):
        logits = self.one_forward(batch)
        return logits

    def accuracy(self, logits, batch, dataset):
        ans_ids = logits.argmax(dim=1).detach().cpu()
        answers = [dataset.answers[ans_id] for ans_id in ans_ids]
        preds = [
            {"question_id": batch["question_id"][i], "answer": answers[i]}
            for i in range(len(logits))
        ]
        accuracy = dataset.evaluate(preds)
        return accuracy

    def forward(self, batch):
        if self.tokenizer is not None:
            question_tokens = self.tokenizer(batch["question"])
            batch["question_tokens"] = question_tokens
        return self.model(batch)["logits"]

    def configure_optimizers(self):
        optim = torch.optim.Adamax(self.parameters())
        return optim