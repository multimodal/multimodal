from multimodal.datasets import VQA, VQA2, VQACP, VQACP2
import tempfile


def test_datasets():
    with tempfile.TemporaryDirectory() as d:
        for dataset in [VQA, VQA2]:
            train = dataset(dir_download=d, split="train", top_answers=3000)
            print(train[0])
            val = dataset(dir_download=d, split="val", top_answers=3000)
            print(val[0])
            test = dataset(dir_download=d, split="test", top_answers=3000)
            print(test[0])
        for dataset in [VQACP, VQACP2]:
            train = dataset(dir_download=d, split="train", top_answers=3000)
            print(train[0])
            test = dataset(dir_download=d, split="test", top_answers=3000)
            print(test[0])

def test_vqa_eval():
    with tempfile.TemporaryDirectory() as d:
        vqa_val = VQA(dir_download=d, split="val", top_answers=3000)
        print(vqa_val[0])
        predicted_answers = [
            {"question_id": qid, "answer": "yes"} for qid in vqa_val.qid_to_annot
        ]
        print(vqa_val.evaluate(predicted_answers))
