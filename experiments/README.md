# Experiments

This directory contains the various research experiments conducted using the `UrbanRepML` library. Each experiment is self-contained in its own subdirectory.

## Experiment Structure

A well-structured experiment should have the following components:

- **`README.md`**: A detailed document explaining the experiment's goals, methodology, data sources, and how to run it.
- **`config.yaml`**: A configuration file that specifies all the parameters for the experiment, such as the study area, H3 resolutions, model hyperparameters, etc.
- **`run_experiment.py` or similar script**: A main script to launch the experiment.
- **`data/`**: A directory to store any data that is generated or specific to this experiment.
- **`scripts/`**: A directory for any helper scripts unique to this experiment.
- **`plots/` or `results/`**: Directories to store the outputs of the experiment, such as visualizations, trained models, and result data.

## How to Create a New Experiment

The best way to create a new experiment is to use the main `run_experiment.py` script from the `scripts/experiments` directory in the repository root.

```bash
python ../scripts/experiments/run_experiment.py \
  --experiment_name my_new_experiment \
  --city <city_name> \
  --fsi_percentile 95
```

This will automatically create a new directory `experiments/my_new_experiment` with the necessary subdirectories and metadata. You can then add a `README.md` and any custom scripts to this new directory.

## Major Experiments

Below is a list of the major experiments in this repository.

### 🌲 `cascadia_exploratory`
- **Status**: ACTIVE
- **Description**: A large-scale experiment focused on the Cascadia bioregion of the Pacific Northwest. It involves multi-year, multi-resolution data processing of satellite embeddings for environmental and land-use analysis.
- **See Also**: `cascadia_exploratory/README.md` for detailed information.

###  archived
- **Status**: ARCHIVED
- **Description**: This directory contains older experiments that are no longer actively being developed but are kept for historical reference.
- **See Also**: The `README.md` files within the individual archived experiment folders.
