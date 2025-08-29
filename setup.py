from setuptools import setup, find_packages
import os

# Read README.md if it exists
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

# Read requirements.txt
requirements = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt', encoding='utf-8') as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]

setup(
    name="urbanrepml",
    version="0.1.0",
    packages=find_packages(exclude=['tests', 'tests.*', 'experiments', 'experiments.*']),
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        'viz': [
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'plotly>=5.14.0',
            'folium>=0.14.0',
            'contextily>=1.3.0',
        ],
        'ml': [
            'wandb>=0.15.0',
            'tensorboard>=2.12.0',
            'optuna>=3.1.0',
        ],
    },
    python_requires=">=3.8",
    author="Bert Berkers",
    author_email="bert.berkers@example.com",
    description="Multi-Modal Urban Representation Learning for Geospatial Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bertberkers/UrbanRepML",
    project_urls={
        "Bug Tracker": "https://github.com/bertberkers/UrbanRepML/issues",
        "Documentation": "https://github.com/bertberkers/UrbanRepML/wiki",
        "Source Code": "https://github.com/bertberkers/UrbanRepML",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="urban-analysis, geospatial, machine-learning, h3, graph-neural-networks, remote-sensing, poi, multi-modal, representation-learning",
    entry_points={
        'console_scripts': [
            'urbanrepml=urban_embedding.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'urbanrepml': [
            'configs/*.yaml',
            'configs/*.yml',
        ],
    },
    zip_safe=False,
)