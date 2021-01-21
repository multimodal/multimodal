from multimodal.datasets import VQA, VQA2, VQACP, VQACP2
import tempfile
import shutil

def test_datasets():
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree("test/data/vqa2", d + "/datasets/vqa2/")
        train = VQA2(dir_data=d, split="train")

        print(train[0])
        val = VQA2(dir_data=d, split="val")
        print(val[0])
        test = VQA2(dir_data=d, split="test")
        print(test[0])
        # test VQA eval
        print(val[0])
        predicted_answers = [
            {"question_id": qid, "answer": "yes"} for qid in val.qid_to_annot
        ]
        print(predicted_answers)
        print(val.evaluate(predicted_answers))


def test_min_ans_occ():
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree("test/data/vqa2", d + "/datasets/vqa2/")
        train = VQA2(dir_data=d, split="train", min_ans_occ=2)
        print(len(train.answers))
