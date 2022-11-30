#! /usr/bin/env python

from setuptools import setup, find_packages
from pathlib import Path

readme = (Path(__file__).parent / 'README.md').read_text()

setup(
    name="deepproblog",
    version="2.0.1",
    description="DeepProbLog: Problog with neural networks",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/ML-KULeuven/deepproblog",
    author="DeepProbLog team",
    author_email="robin.manhaeve@cs.kuleuven.be",
    license="Apache Software License",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Prolog",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="prolog probabilistic logic neural-symbolic problog deepproblog",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
    include_package_data=True,
    install_requires=["pysdd",
                      "problog",
                      "torch",
                      "torchvision"],
    extras_require={
        "approximate": ["pyswip @ git+https://github.com/ML-KULeuven/pyswip.git#egg=pyswip"],
        "examples": ["Pillow"],
        "tests": ["pytest"],
    },
)
