from setuptools import setup, find_packages

setup(
    name="multimodal_preprocessing",
    version="0.1.0",
    description="A library for multimodal data preprocessing",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "spacy",
        "imbalanced-learn",
    ],
    python_requires=">=3.7",
)
