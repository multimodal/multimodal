from multimodal.datasets import VQA, VQA2, VQACP, VQACP2
import tempfile


def test_vqa():
    with tempfile.TemporaryDirectory() as d:
        vqa_train = VQA(dir_download=d, split="train", top_answers=3000)
        print(vqa_train[0])
        vqa_val = VQA(dir_download=d, split="val", top_answers=3000)
        print(vqa_val[0])
        vqa_test = VQA(dir_download=d, split="test", top_answers=3000)
        print(vqa_test[0])


def test_vqa_eval():
    with tempfile.TemporaryDirectory() as d:
        vqa_val = VQA(dir_download=d, split="val", top_answers=3000)
        print(vqa_val[0])
        predicted_answers = [
            {"question_id": qid, "answer": "yes"} for qid in vqa_val.qid_to_annot
        ]
        print(vqa_val.evaluate(predicted_answers))
