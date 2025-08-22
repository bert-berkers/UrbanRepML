# UrbanRepML & Cascadia Experiment 🏙️🌲

This repository contains the `UrbanRepML` library for multi-scale urban representation learning and the `Cascadia` experiment, a large-scale application of this technology to study the bioregions of the Pacific Northwest.

## 🚀 Getting Started

First, clone the repository and install the required dependencies in editable mode. This command works for both the core library and the experiments.

```bash
# Clone the repository
git clone https://github.com/yourusername/UrbanRepML.git
cd UrbanRepML

# Install the package and dependencies
pip install -e .
```

For AI agent collaborators (like Jules and Claude), please see **`AGENTS.md`** for standardized workflows and commands.

---

## 🔬 The `UrbanRepML` Library

`UrbanRepML` is a Python library that learns meaningful representations of urban areas by integrating multiple data sources (transit, roads, buildings, points of interest) across different spatial scales. It uses a novel U-Net-inspired GNN architecture to capture how cities function at walking, cycling, and driving scales simultaneously.

### 🎯 Core Features
- **Hierarchical Analysis**: Creates and processes multi-scale H3 hexagonal grids.
- **Multi-Modal Integration**: Fuses data from GTFS, OpenStreetMap, satellite imagery, and more.
- **Representation Learning**: Trains a powerful GNN to produce rich, dense embeddings of urban areas.
- **Flexible & Modular**: Designed to be configurable and extensible for new cities and data sources.

### 🔧 Running a `UrbanRepML` Experiment
You can run a complete experiment using the central `run_experiment.py` script:

```bash
python scripts/experiments/run_experiment.py \
  --experiment_name amsterdam_dense \
  --city amsterdam \
  --fsi_percentile 90 \
  --run_training \
  --epochs 200
```

For more details on the library's architecture and capabilities, see the **`urban_embedding/README.md`** file.

---

## 🌲 The Cascadia Experiment

The Cascadia experiment is a large-scale research project applying the `UrbanRepML` toolkit to the Cascadia bioregion (covering Oregon and Northern California). Its goal is to create a detailed, multi-resolution, multi-year dataset of environmental and land-use embeddings for analysis and integration with the [GEO-INFER](https://github.com/ActiveInferenceInstitute/GEO-INFER) project.

### 🎯 Core Features
- **Large Geographic Scope**: Covers over 400,000 km² across two states.
- **Multi-Resolution**: Processes data from H3 resolutions 5 (regional) to 11 (ultra-fine).
- **Temporal Analysis**: Incorporates satellite data from 2017-2024.
- **Advanced Processing**: Includes a two-stage parallel processing pipeline for handling massive amounts of data.

### 🔧 Running the Cascadia Processing Pipeline
The Cascadia experiment has its own specialized workflow.

```bash
# Navigate to the experiment directory
cd experiments/cascadia_exploratory

# Run the two-stage processing pipeline
python run_coastal_processing.py --workers 6
python stitch_results.py
```
For complete instructions, see the **`experiments/cascadia_exploratory/README.md`** file.

---

## 📁 Project Structure

This repository is organized into a core library and a set of experiments.

```
UrbanRepML/
├── AGENTS.md                # Standardized instructions for AI agents
├── urban_embedding/         # The core UrbanRepML library
│   └── README.md            # --> Detailed library documentation
├── experiments/             # Research experiments applying the library
│   ├── README.md            # --> Guide to the experiments structure
│   └── cascadia_exploratory/
│       └── README.md        # --> Detailed Cascadia experiment guide
├── data/                    # Data used for experiments
│   └── README.md            # --> Overview of data sources and structure
├── scripts/                 # Reusable scripts for data processing and experiments
└── DEVELOPMENT_LOG.md       # A narrative log of development sessions
```

## 📚 Documentation

- **`AGENTS.md`**: Standardized instructions for AI agents (Jules & Claude).
- **`ARCHITECTURE.md`**: The deep-dive technical design of the `UrbanRepML` system.
- **`CONFIG_GUIDE.md`**: A reference for all configuration parameters.
- **`DEVELOPMENT_LOG.md`**: The narrative history of the project's development.

## 🤝 Contributing

We welcome contributions! Please add your work sessions to the `DEVELOPMENT_LOG.md` to help us maintain a clear project narrative.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{urbanrepml2025,
  title = {UrbanRepML: Multi-scale Urban Representation Learning and the Cascadia Bioregional Experiment},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/UrbanRepML}
}
```