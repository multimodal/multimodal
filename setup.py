import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PKGS = [
    "numpy",
    "tqdm",
    "appdirs",
    "torch>=1",
    "pySmartDL",
    "torchtext>=0.6",
    "tables",
    "Pillow>=6",
    "torchvision",
]


setuptools.setup(
    name="multimodal",  # Replace with your own username
    version="0.0.11",
    author="Corentin Dancette",
    author_email="corentin@cdancette.fr",
    description="A collection of multimodal datasets multimodal for research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cdancette/multimodal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=REQUIRED_PKGS,
)
