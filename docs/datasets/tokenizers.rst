Visual Question Answering Tokenizers
------------------------------------

For VQA, the tokenizer we implement is a simple tokenizer based on torchtext basic_english.

The class you need to use is the `BasicTokenizer` class.

.. code-block:: python

   from multimodal.text import BasicTokenizer
   tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa")
   tokens = tokenizer("What color is the car?")
   # feed tokens to model


.. autoclass:: multimodal.text.BasicTokenizer


