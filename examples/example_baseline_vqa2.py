# training a simple baseline on VQA V2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from multimodal.datasets import VQA2
from multimodal.text import WordEmbedding
from statistics import mean
from torch.nn.utils.weight_norm import weight_norm


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, dropout):
        """Module for question embedding
        """
        super().__init__()

        self.rnn = nn.GRU(
            in_dim,
            num_hid,
            nlayers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
        )
        self.num_hid = num_hid
        self.nlayers = nlayers

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers, batch, self.num_hid)
        return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output[:, -1]

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super().__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class SimpleVQAModel(pl.LightningModule):
    def __init__(self, answers, train_dataset=None, val_dataset=None):
        super().__init__()

        self.answers = answers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.wemb = WordEmbedding.from_pretrained(
            "glove.6B.100d", freeze=True, max_tokens=3000
        )

        dim_v = 2048
        dim_q = 1024
        num_hid = 2048
        dim_word_emb = 100

        self.q_emb = QuestionEmbedding(dim_word_emb, dim_q, nlayers=1, dropout=0.0)
        # self.v_att = Attention(dim_v, dim_q, num_hid)
        self.q_net = FCNet([dim_q, num_hid])
        self.v_net = FCNet([dim_v, num_hid])

        self.classifier = nn.Sequential(
            weight_norm(nn.Linear(num_hid, 2 * num_hid), dim=None),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=True),
            weight_norm(nn.Linear(2 * num_hid, len(self.answers)), dim=None),
        )

    def forward(self, batch):

        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        v = batch["features"]  # N, num_feat, dim
        question = batch["question"]

        w_emb = self.wemb(question)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        # att = self.v_att(v, q_emb)
        # v_emb = (att * v).sum(1)  # [batch, v_dim]
        v_emb = v.mean(1)  # (batch, v_dim)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        # Get answers from logits
        ans_ids = logits.argmax(dim=1).detach().cpu()
        answers = [self.train_dataset.answers[ans_id] for ans_id in ans_ids]
        preds = [
            {"question_id": batch["question_id"][i], "answer": answers[i]}
            for i in range(len(question))
        ]
        return logits, preds

    def training_step(self, batch, batch_idx):
        logits, preds = self.forward(batch)
        label = batch["label"]
        loss = F.binary_cross_entropy_with_logits(logits, label)
        self.log("Loss/train_loss", loss)
        scores = scores = self.train_dataset.evaluate(preds)
        self.log(
            "Accuracy/train_acc",
            scores["overall"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "preds": preds}

    def training_epoch_end(self, training_step_outputs):
        all_preds = []
        for out in training_step_outputs:
            all_preds.extend(out["preds"])
        scores = self.train_dataset.evaluate(all_preds)
        for key in scores:
            self.log(f"Accuracy/train_acc_{key}", scores[key])

    def validation_step(self, batch, batch_idx):
        logits, preds = self.forward(batch)
        label = batch["label"]
        loss = F.binary_cross_entropy_with_logits(logits, label)
        self.log("Loss/val_loss", loss)
        scores = scores = self.val_dataset.evaluate(preds)
        self.log(
            "Accuracy/val_acc",
            scores["overall"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "preds": preds}

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = []
        for out in validation_step_outputs:
            all_preds.extend(out["preds"])
        scores = self.val_dataset.evaluate(all_preds)
        for key in scores:
            self.log(f"Accuracy/val_acc_{key}", scores[key])

    def configure_optimizers(self):
        return torch.optim.Adamax(self.parameters())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-data")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    args = parser.parse_args()

    # Warning: those two lines will download COCO bottom-up features which
    # are big files. This might take a long time.
    vqa_train = VQA2(
        split="train",
        features="coco-bottom-up-36",
        dir_data=args.dir_data,
        min_ans_occ=10,
    )

    vqa_val = VQA2(
        split="val",
        features="coco-bottom-up-36",
        dir_data=args.dir_data,
        min_ans_occ=10,
    )

    model = SimpleVQAModel(
        answers=vqa_train.answers, train_dataset=vqa_train, val_dataset=vqa_val
    )
    trainer = pl.Trainer(gpus=1, default_root_dir=args.root_dir,)

    trainer.fit(
        model,
        DataLoader(
            vqa_train,
            batch_size=args.batch_size,
            collate_fn=VQA2.collate_fn,
            num_workers=args.num_workers,
            shuffle=True,
        ),
        DataLoader(
            vqa_val,
            batch_size=args.batch_size,
            collate_fn=VQA2.collate_fn,
            num_workers=args.num_workers,
        ),
    )
