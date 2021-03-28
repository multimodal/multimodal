Command Line Interface
----------------------

All commands start with :code:`python -m multimodal <subcommand>`

The subcommands available are listed here:

VQA Evaluation: ``vqa-eval``
==============

Description
+++++++++++
Run the evaluation, following the VQA evaluation metric, taking into account answers from multiple humans.


.. code-block:: bash
    
    python -m multimodal vqa-eval -p <predictions-path> -s <split> --dir_data <multimodal_dir_data>


Options
+++++++

.. program:: python -m multimodal vqa-eval

.. option:: -p <path>, --predictions <path>
    
    path to predictions, should follow the official VQA evaluation format (see https://visualqa.org/evaluation.html)

.. option:: -s <split>, --split <split>
    
    VQA split, either :code:`train`, :code:`val` or :code:`test`
    depending on the dataset (in VQA-CP, there are only train and test).

.. option:: --dir_data <dir_data> (optional)
    
    path where data will be downloaded if necessary. By default in appdata.

Example

.. code-block:: bash
    
    $ python -m multimodal vqa-eval -s val -p logs/updown/predictions.json
    Loading questions
    Loading annotations
    Loading aid_to_ans
    {'overall': 0.6346422273435531, 'yes/no': 0.8100979625284017, 'number': 0.42431932892585483, 'other': 0.5569148080507953}


Data Download: ``download``
=============

Description
+++++++++++


Download and process data.

.. code-block::

    python -m multimodal download <dataset> --dir_data <dir_data>

Options
+++++++

.. program:: python -m multimodal download

.. option:: --dir_data <dir_data> (optional)
    
    path where data will be downloaded if necessary. By default in appdata.

.. option:: dataset

    Name of the dataset to download. 
    Can be either ``VQA``, ``VQA2``, ``VQACP``, ``VQACP2``, ``coco-bottom-up``, ``coco-bottomup-36``.