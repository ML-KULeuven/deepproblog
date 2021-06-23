#! /usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="deepproblog",
    version="2.0.0",
    description="DeepProbLog: Problog with neural networks",
    url="https://dtai.cs.kuleuven.be/problog",
    author="DeepProbLog team",
    author_email="robin.manhaeve@cs.kuleuven.be",
    license="Apache Software License",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Prolog",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="prolog probabilistic logic neural-symbolic problog deepproblog",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "problog",
        "pyswip",
        "torch",
        "torchvision",
        "pysdd",
    ],
    extras_require={
        "examples": ["networkx", "Pillow", "scipy", "scikit-learn"],
        "tests": ["pytest"],
    },
)
