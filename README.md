# Evolving Curricula with Regret-Based Environment Design

Implementation of the paper "Evolving Curricula with Regret-Based Environment Design".


## Installation

To install the environment, install the `miniconda` package manager, open the Anaconda prompt and run the following command:

```bash
conda env create -f accel_env.yaml
```

This will create a new conda environment called `accel` with all the necessary dependencies. To activate the environment, run:

```bash
conda activate accel
```

Or, if you are using a Jupiter notebook in VSCode, select the `accel` environment in the top right corner of the notebook.

> If you added some new dependencies to the environment, you can update the `accel_env.yaml` file by running:
> ```bash
> conda env create -f accel_env.yaml
> ```


## Dependencies

- Stable Baselines3: https://stable-baselines3.readthedocs.io/en/master/
- Minigrid documentation: https://minigrid.farama.org/