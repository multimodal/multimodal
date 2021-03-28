import argparse
import json
from multimodal.datasets.vqa import VQA, VQA2, VQACP, VQACP2
from multimodal import DEFAULT_DATA_DIR


class VQAEvalCommand:
    dataset = VQA
    command = "vqa-eval"

    @classmethod
    def add_parser(cls, subparser):
        parser: argparse.ArgumentParser = subparser.add_parser(cls.command)
        parser.add_argument("--dir_data", default=DEFAULT_DATA_DIR)
        parser.add_argument(
            "-p", "--predictions", help="path to predictions", required=True
        )
        parser.add_argument(
            "-s", "--split", default="train", choices=["train", "val", "test"], required=True
        )
        parser.set_defaults(func=cls.run)

    @classmethod
    def run(cls, args):
        dataset = cls.dataset(dir_data=args.dir_data, split=args.split)
        with open(args.predictions) as f:
            predictions = json.load(f)
        result = dataset.evaluate(predictions)
        print(result)


class VQA2EvalCommand(VQAEvalCommand):
    dataset = VQA2
    command = "vqa2-eval"


class VQACPEvalCommand(VQAEvalCommand):
    dataset = VQACP
    command = "vqacp-eval"


class VQACP2EvalCommand(VQAEvalCommand):
    dataset = VQACP2
    command = "vqacp2-eval"
