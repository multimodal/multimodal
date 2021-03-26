import argparse
from multimodal.commands.vqa_eval import VQA2EvalCommand, VQACP2EvalCommand, VQACPEvalCommand, VQAEvalCommand

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")
subparsers.required = True

for cls in [
    VQAEvalCommand,
    VQA2EvalCommand,
    VQACPEvalCommand,
    VQACP2EvalCommand,
]:
    cls.add_parser(subparsers)

args = parser.parse_args()
args.func(args)


