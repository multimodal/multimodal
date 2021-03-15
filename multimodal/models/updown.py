from multimodal.text.wordembedding import PretrainedWordEmbedding
from multimodal.text.basic_tokenizer import BasicTokenizer
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        self.rnn = nn.GRU(
            in_dim,
            num_hid,
            nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True,
        )

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, : self.num_hid]
        backward = output[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


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
        super(Attention, self).__init__()
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


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class UpDownModel(nn.Module):
    """
    The UpDown / BUTD model (Bottom-Up and Top-Down attention) by 
    Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang.

    Paper: https://arxiv.org/abs/1707.07998

    The implementation was adapted from this code: https://github.com/hengyuan-hu/bottom-up-attention-vqa.
    """
    def __init__(
        self,
        num_ans,
        v_dim=2048,
        num_hid=2048,
        new_attention=True,
        tokens: list = None,
        num_tokens=None,
        padding_idx=None,
        freeze_emb=False,
    ):
        super().__init__()
        if tokens is not None:
            self.w_emb = PretrainedWordEmbedding(
                "glove.6B.300d", tokens=tokens, freeze=freeze_emb, padding_idx=None
            )
        else:
            self.w_emb = nn.Embedding(num_tokens, 300, padding_idx=padding_idx)
            if freeze_emb:
                self.w_emb.weight.requires_grad = False

        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        if new_attention:
            self.v_att = NewAttention(v_dim, self.q_emb.num_hid, num_hid)
        else:
            self.v_att = Attention(v_dim, self.q_emb.num_hid, num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, num_hid])
        self.v_net = FCNet([v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, num_ans, 0.5)

    def forward(self, batch):
        """
        Forward method

        Args:
            batch (dict): Dictionnary containing the keys ``features`` (B*N*D tensor) and``question_tokens`` (B*L tensor)
                representing respectively the image and the question.
        
        Returns:
            (dict) containing the key ``logits``, which has dimension ``(batch, num_ans)`` containing the unnormalized logits.
        """
        v = batch["features"]
        q = batch["question_tokens"]
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        out = {
            "logits": logits,
            # "mm": joint_repr,
            # "processed_question": q_emb,
        }
        return out

    @classmethod
    def from_pretrained(cls, name="updown-base"):
        """
        One of "updown-base-100, updown-base-36, updown-newatt-100, updown-newatt-36
        """
        pass


if __name__ == "__main__":
    import argparse
    import pytorch_lightning as pl
    from multimodal.models.lightning import VQALightningModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-data", help="dir where multimodal data (datasets, features) will be stored")
    parser.add_argument("--dir-exp", default="logs/vqa2/updown")
    parser.add_argument("--v_dim", type=int, default=2048)
    parser.add_argument("--num_hid", type=int, default=2048)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--min-ans-occ", default=9, type=int)
    parser.add_argument("--features", default="coco-bottomup-36")
    parser.add_argument("--old-attention", action="store_true")
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--clip_grad", type=float, default=0.25)
    parser.add_argument("--freeze_emb", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpus", help="Number of gpus to use")
    parser.add_argument("--distributed-backend")

    args = parser.parse_args()

    from multimodal.datasets.lightning import VQA2DataModule
    tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa2", dir_data=args.dir_data)

    vqa2 = VQA2DataModule(
        dir_data=args.dir_data,
        min_ans_occ=args.min_ans_occ,
        features=args.features,
        label="multilabel",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    vqa2.prepare_data()
    vqa2.setup()

    num_ans = len(vqa2.train_dataset.answers)
    updown = UpDownModel(
        num_ans=num_ans,
        v_dim=args.v_dim,
        num_hid=args.num_hid,
        tokens=tokenizer.tokens,
        padding_idx=tokenizer.pad_token_id,
        new_attention=not args.old_attention,
        freeze_emb=args.freeze_emb
    )

    lightningmodel = VQALightningModule(
        updown,
        train_dataset=vqa2.train_dataset,
        val_dataset=vqa2.val_dataset,
        tokenizer=tokenizer,
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        gradient_clip_val=args.clip_grad,
        default_root_dir=args.dir_exp,
        profiler="simple",
        distributed_backend=args.distributed_backend,
    )

    trainer.fit(lightningmodel, datamodule=vqa2)
