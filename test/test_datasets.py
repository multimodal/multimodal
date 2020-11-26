from multimodal.datasets import VQA, VQA2, VQACP, VQACP2
import tempfile

def test_full_download():
    with tempfile.TemporaryDirectory() as d:
        dataset = VQA(dir_data=d, split="val")

def test_datasets():
    with tempfile.TemporaryDirectory() as d:
        for dataset in [VQA, VQA2]:
            train = dataset(dir_data=d, split="train")
            print(train[0])
            val = dataset(dir_data=d, split="val")
            print(val[0])
            test = dataset(dir_data=d, split="test")
            print(test[0])

def test_datasets_cp():
    with tempfile.TemporaryDirectory() as d:
        for dataset in [VQACP, VQACP2]:
            train = dataset(dir_data=d, split="train")
            print(train[0])
            test = dataset(dir_data=d, split="test")
            print(test[0])

def test_vqa_eval():
    with tempfile.TemporaryDirectory() as d:
        vqa_val = VQA(dir_data=d, split="val")
        print(vqa_val[0])
        predicted_answers = [
            {"question_id": qid, "answer": "yes"} for qid in vqa_val.qid_to_annot
        ]
        print(vqa_val.evaluate(predicted_answers))
