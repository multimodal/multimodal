from multimodal.datasets import VQA, VQA2, VQACP, VQACP2
import tempfile

def test_datasets():
    with tempfile.TemporaryDirectory() as d:
        train = VQA(dir_data=d, split="train")
        print(train[0])
        val = VQA(dir_data=d, split="val")
        print(val[0])
        test = VQA(dir_data=d, split="test")
        print(test[0])

        # test VQA eval
        print(val[0])
        predicted_answers = [
            {"question_id": qid, "answer": "yes"} for qid in val.qid_to_annot
        ]
        print(val.evaluate(predicted_answers))


def test_min_ans_occ():
    with tempfile.TemporaryDirectory() as d:
        train = VQA2(dir_data=d, split="train", min_ans_occ=8)
        print(len(train.answers))
