# DeepProbLog

[![Python application](https://github.com/ML-KULeuven/deepproblog-dev/actions/workflows/python-app.yml/badge.svg)](https://github.com/ML-KULeuven/deepproblog-dev/actions/workflows/python-app.yml)

DeepProbLog is an extension of [ProbLog](https://dtai.cs.kuleuven.be/problog/)
that integrates Probabilistic Logic Programming with deep learning by introducing the neural predicate. 
The neural predicate represents probabilistic facts whose probabilites are parameterized by neural networks.
For more information, consult the papers listed below.


## Requirements

DeepProbLog has the following requirements:

* [ProbLog](https://dtai.cs.kuleuven.be/problog/)
* [PySDD](https://pysdd.readthedocs.io/en/latest/)
* [PyTorch](https://pytorch.org/)
* [TorchVision](https://pytorch.org/vision/stable/index.html)
* [PySwip](https://github.com/yuce/pyswip)

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
