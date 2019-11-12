from setuptools import find_packages, setup

setup(
    name="pytorch-bert",
    version="1.0.0a1",
    install_requires=["torch>=1.2.0"],
    extras_require={"with-tf": ["tensorflow>=2.0.0", "numpy"]},
    packages=find_packages(exclude=["tests"]),

    description="bert implementation",
    author="Jeong Ukjae",
    author_email="jeongukjae@gmail.com",
    url="https://github.com/jeongukjae/pytorch-bert",
    python_requires=">=3.6, <3.8",
)
