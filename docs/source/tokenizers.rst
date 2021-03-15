Tokenizers
##########

Base class
----------

We implement a basic tokenizer class, built on torchtext basic_english tokenizer.
Its purpose is to transform text into a series of token_ids (integers), that will be fed to 
a word vector.

.. autoclass:: multimodal.text.BasicTokenizer


VQA v2
------

The pretrained tokenizer for VQA v2 is called :code:`pretrained-vqa2`.


.. code-block:: python

   from multimodal.text import BasicTokenizer
   tokenizer = BasicTokenizer.from_pretrained("pretrained-vqa2")
   tokens = tokenizer("What color is the car?")
   # feed tokens to model


