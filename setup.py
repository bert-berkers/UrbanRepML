from setuptools import setup, find_packages
import os

# Read README.md if it exists
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="urbanrepml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "pandas",
        "numpy",
        "geopandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "wandb",
        "osmnx",
        "networkx",
        "h3",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Urban representation learning with multi-level analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/UrbanRepML",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)