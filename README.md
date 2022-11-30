# DeepProbLog
[![Unit tests](https://github.com/ML-KULeuven/deepproblog/actions/workflows/python-app.yml/badge.svg)](https://github.com/ML-KULeuven/deepproblog/actions/workflows/python-app.yml)

DeepProbLog is an extension of [ProbLog](https://dtai.cs.kuleuven.be/problog/)
that integrates Probabilistic Logic Programming with deep learning by introducing the neural predicate. 
The neural predicate represents probabilistic facts whose probabilites are parameterized by neural networks.
For more information, consult the papers listed below.

## Installation
DeepProbLog can easily be installed using the following command:

`pip install git+https://github.com/ML-Kuleuven/deepproblog.git`

## Test
To make sure your installation works, install pytest `pip install pytest`, and run `pytest .` inside of the deepproblog directory.

## Troubleshooting
In some cases, the installation of PySDD can fail, as indicated by the following error:
```
problog.errors.InstallError: The SDD library is not available. Please install the PySDD package.
```
To fix this, reinstall PySDD from source:
```
pip uninstall pysdd
pip install git+https://github.com/wannesm/PySDD.git#egg=PySDD
```

## Requirements

DeepProbLog has the following requirements:
* Python > 3.9
* [ProbLog](https://dtai.cs.kuleuven.be/problog/)
* [PySDD](https://pysdd.readthedocs.io/en/latest/)
    - Use `pip install git+https://github.com/wannesm/PySDD.git#egg=PySDD`
* [PyTorch](https://pytorch.org/)
* [TorchVision](https://pytorch.org/vision/stable/index.html)

## Approximate Inference
To use Approximate Inference, we have the followign additional requirements
* [PySwip](https://github.com/ML-KULeuven/pyswip) 
    - Use `pip3 install git+https://github.com/ML-KULeuven/pyswip`
* [SWI-Prolog < 9.0.0](https://www.swi-prolog.org/)
The latter can be installed on Ubuntu with the following commands:
```
sudo apt-add-repository ppa:swi-prolog/stable
sudo apt install swi-prolog=8.4* swi-prolog-nox=8.4* swi-prolog-x=8.4*
```
## Experiments

The experiments are presented in the papers are available in the [src/deepproblog/examples](src/deepproblog/examples) directory.

## Papers
1. Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, Luc De Raedt:
*DeepProbLog: Neural Probabilistic Logic Programming*. NeurIPS 2018: 3753-3763 ([paper](https://papers.nips.cc/paper/2018/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html))
2. Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, Luc De Raedt:
*Neural Probabilistic Logic Programming in DeepProbLog*. AIJ ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0004370221000552))
3. Robin Manhaeve, Giuseppe Marra, Luc De Raedt:
*Approximate Inference for Neural Probabilistic Logic Programming*. KR 2021
## License
Copyright 2021 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
