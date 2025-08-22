# AI Agent Instructions

This document provides guidance for AI agents (like Jules and Claude) working in the UrbanRepML repository.

## 🚀 Core Commands

### Environment Setup
To set up your local environment and install the necessary dependencies, run the following command from the root of the repository:
```bash
pip install -e .
```

### Running Tests
To ensure the codebase is functioning correctly, you can run the test suite.
```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests/test_pipeline.py
```

### Running the Experiment Pipeline
The primary way to run experiments is through the `run_experiment.py` script. This script orchestrates the entire workflow from data preprocessing to model training.

#### Generic Experiment Execution
```bash
python scripts/experiments/run_experiment.py \
  --experiment_name <your_experiment_name> \
  --city <city_name> \
  --fsi_percentile <percentage> \
  --run_training \
  --epochs <number_of_epochs>
```
*   `--experiment_name`: A unique name for your experiment. Outputs will be saved in `experiments/<your_experiment_name>/`.
*   `--city`: The name of the city or region to process.
*   `--fsi_percentile`: The floor space index percentile to use for filtering dense urban areas (e.g., 95 for the top 5%).
*   `--run_training`: A flag to indicate that the model should be trained.
*   `--epochs`: The number of training epochs.

## 🏛️ Architecture Overview

This project learns urban representations at multiple spatial resolutions using a Graph Neural Network (GNN).

### Core Components
1.  **Pipeline System** (`urban_embedding/pipeline.py`): Orchestrates the entire workflow.
2.  **Feature Processing** (`urban_embedding/feature_processing.py`): Processes multimodal urban data (GTFS, road networks, aerial imagery, POIs).
3.  **Graph Construction** (`urban_embedding/graph_construction.py`): Builds spatial graphs based on travel time accessibility.
4.  **Model Architecture** (`urban_embedding/model.py`): Implements the UrbanUNet, a U-Net style GNN.
5.  **Analytics** (`urban_embedding/analytics.py`): Handles embedding visualization and clustering.

### Data Flow
1.  **Preprocessing**: Raw urban data → H3 hexagonal regions with building density.
2.  **Feature Extraction**: Multimodal embeddings → PCA-reduced features.
3.  **Graph Building**: Spatial accessibility calculations → Weighted graphs.
4.  **Model Training**: Multi-resolution GNN training → Urban embeddings.
5.  **Analysis**: Embeddings → Clustering and visualization.

Refer to `ARCHITECTURE.md` for a more detailed technical breakdown.

## 📝 Development Log & Narrative Account

To maintain a clear narrative of the project's history and progress, all significant work sessions must be logged in `DEVELOPMENT_LOG.md`.

When you complete a task or a work session, please add a new entry to the top of this file. Each entry should include:
*   The date.
*   A clear heading describing the goal of the session.
*   A summary of the problems addressed and the accomplishments.
*   Any technical details, new scripts, or configuration changes.

This log serves as our shared "memory" and is crucial for collaboration.
