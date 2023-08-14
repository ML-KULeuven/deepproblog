Installing DeepProbLog
======================

Installation
------------
DeepProbLog can easily be installed using the following command:
Make sure the following packages are installed:

.. code-block:: bash

    pip install deepproblog

Test
----
To make sure your installation works, install pytest

.. code-block:: bash

    pip install pytest

and run

.. code-block:: bash

    python -m deepproblog test


Requirements
------------

DeepProbLog has the following requirements:

* Python > 3.9
* [ProbLog](https://dtai.cs.kuleuven.be/problog/)
* [PySDD](https://pysdd.readthedocs.io/en/latest/)
* [PyTorch](https://pytorch.org/)
* [TorchVision](https://pytorch.org/vision/stable/index.html)

Approximate Inference
---------------------
To use Approximate Inference, we have the followign additional requirements

* [PySwip](https://github.com/ML-KULeuven/pyswip)

.. code-block::

    pip install git+https://github.com/ML-KULeuven/pyswip

* [SWI-Prolog < 9.0.0](https://www.swi-prolog.org/)

The latter can be installed on Ubuntu with the following commands:

.. code-block:: bash

  sudo apt-add-repository ppa:swi-prolog/stable
  sudo apt install swi-prolog=8.4* swi-prolog-nox=8.4* swi-prolog-x=8.4*
