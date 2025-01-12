from setuptools import setup, find_packages

setup(
    name="multimodal_preprocessing",  # Unique package name
    version="0.1.0",  # Initial version
    description="A library for preprocessing multimodal data (text, numeric, video).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your_email@example.com",
    url="https://github.com/your-username/multimodal-preprocessing",  # GitHub repo URL
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "spacy",
        "imbalanced-learn",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
