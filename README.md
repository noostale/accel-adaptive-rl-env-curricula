# Evolving Curricula with Regret-Based Environment Design

https://prod.liveshare.vsengsaas.visualstudio.com/join?80FAEE665B8434DF404E4133482D9B7EA903

Implementation of the paper "Evolving Curricula with Regret-Based Environment Design".

## Installation

To install the environment, install the `miniconda` package manager, open the Anaconda prompt and run the following command:

```bash
conda env create --file accel_env.yaml # cpu-only
conda env create --file accel_env_cuda.yaml # with cuda
```

This will create a new conda environment called `accel`  or `accel_env_cuda` with all the necessary dependencies.

If you already installed the environment and want to update it, run:

```bash
conda env update --file accel_env.yaml --prune # cpu-only
conda env update --file accel_env_cuda.yaml --prune # with cuda
```

To activate the environment, run:

```bash
conda activate accel # cpu-only
conda activate accel_cuda # with cuda
```

Or, if you are using a Jupiter notebook in VSCode, select the `accel` environment in the top right corner of the notebook.

> If you added some new dependencies to the environment, you can update the `accel_env.yaml` file by running:
>
> ```bash
> conda env export > accel_env.yaml # cpu-only
> conda env export > accel_env_cuda.yaml # with cuda
> ```

## Dependencies

- Gymnasium: https://gym.openai.com/
- Stable Baselines3: https://stable-baselines3.readthedocs.io/en/master/
- Minigrid documentation: https://minigrid.farama.org/
- Wandb: https://wandb.ai/
