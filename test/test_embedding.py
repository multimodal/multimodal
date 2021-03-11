from multimodal.text import PretrainedWordEmbeddingWithTokenizer
import torch

def test_embedding():

    w = PretrainedWordEmbeddingWithTokenizer(tokens=["hello", "hi", "this", "is", "a", "test"], dim=10)

    sentence1 = "Hello this is a test"

    out = w([sentence1])
    assert out.shape == torch.Size([1, 5, 10])

    # test unknown tokens
    unknown = "tokens are unknown"
    out = w([unknown])
    # breakpoint()
    assert torch.all(out[0, 0] == out[0, 1]).item()
    assert torch.all(out[0, 0] == out[0, 2]).item()

    # test padding
    sentences = [
        "Hello this is a test",
        "hello this is a test with padding",
    ]

    out = w(sentences)
    assert out.shape ==  torch.Size([2, 7, 10])
