from setuptools import find_packages, setup

setup(
    name="pytorch-bert",
    version="1.0.0a2",
    install_requires=["torch>=1.3.0"],
    extras_require={"with-tf": ["tensorflow>=2.0.0", "numpy"]},
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6, <3.8",
    #
    description="bert implementation",
    author="Jeong Ukjae",
    author_email="jeongukjae@gmail.com",
    url="https://github.com/jeongukjae/pytorch-bert",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
