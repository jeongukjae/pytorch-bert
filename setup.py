from setuptools import find_packages, setup

setup(
    name="pytorch-bert",
    version="0.0.1",
    description="bert implementation",
    install_requires=["torch"],
    url="https://github.com/jeongukjae/pytorch-bert",
    author="Jeong Ukjae",
    author_email="jeongukjae@gmail.com",
    packages=find_packages(exclude=["tests"]),
)
