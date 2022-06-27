import os
from setuptools import find_packages, setup

__version__ = None


# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="NLP_barriers",
    version=__version__,
    author="Kevin Spiekermann",
    description="Uses transformers to predict reaction properties.",
    url="https://github.com/kspieks/NLP_barriers",
    packages=find_packages(),
    long_description=read('README.md'),
)
