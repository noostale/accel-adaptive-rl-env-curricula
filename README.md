# ACCEL - Evolving Curricula with Regret-Based Environment Design

This repository contains the code for an implementation of the paper **[Evolving Curricula with Regret-Based Environment Design](https://arxiv.org/abs/2203.01302)** by Jack Parker-Holder, Minqi Jiang, and Michael Dennis, presented at ICML 2022.

| **Method**                                                                | **Visualization**                                                 |
|---------------------------------------------------------------------------|-------------------------------------------------------------------|
| **ACCEL + EasyStart**<br> ![ACCEL + EasyStart](gifs/level_accel_easy.gif) | **ACCEL**<br> ![ACCEL](gifs/level_accel.gif)                      |
| **DR (Domain Randomization)**<br> ![DR](gifs/level_dr.gif)                | **PLR (Prioritized Level Replay)**<br> ![PLR](gifs/level_plr.gif) |

## Overview
ACCEL (Adversarially Compounding Complexity by Editing Levels) is an evolutionary approach to Unsupervised Environment Design (UED) in reinforcement learning. It leverages regret-based objectives to dynamically adjust training environments, fostering systematic generalization and robust agent training.

## Introduction
Training reinforcement learning agents in complex environments remains a challenging task due to the high-dimensional state spaces.

Adaptive curricula have shown promise in improving agent robustness by dynamically adjusting training environments. Unsupervised Environment Design (UED) formalizes this approach, allowing agents to generalize systematically.

### Our Approach: ACCEL
ACCEL is an evolutionary algorithm that modifies levels in reinforcement learning environments using regret-based objectives. It progressively increases complexity by selecting and editing high-regret levels iteratively. New levels are randomly generated to maintain diversity and allow exploration.

**Key Features:**
- Uses **regret-based objectives** to modify training levels.
- **Edits high-regret levels** iteratively for progressively harder challenges.
- **Randomly generates levels** to promote exploration.
- Ensures **systematic generalization** of the learning process.

## Methodology
### ACCEL Algorithm Workflow
1. **Select high-regret levels** from the replay buffer.
2. **Modify selected levels** by adding one block at a time.
3. **Add or discard levels** based on regret values:
   - If regret > 0 → Level is added to the level buffer.
   - If regret = 0 → Level is discarded (either too easy or too difficult).
4. **Evaluate and replay** the modified levels to continue training.

### Environment Details
- **Framework:** MiniGrid
- **Setup:** Partially observable grids with obstacles, goals, and navigable spaces.
- **Task:** Agent must navigate to the goal while overcoming increasing complexity.
- **Rewards:** Sparse, based on reaching the goal within fewer steps.

## Comparisons with Other UED Approaches
ACCEL was compared against other UED strategies:

| Method                             | Description                                                  | Limitations                                          |
|------------------------------------|--------------------------------------------------------------|------------------------------------------------------|
| **DR (Domain Randomization)**      | Randomly samples environments for diverse training           | Lacks focus on agent-specific weaknesses             |
| **PLR (Prioritized Level Replay)** | Replays high-regret levels to improve robustness             | Does not modify levels dynamically                   |
| **ACCEL**                          | Mutates high-regret levels to increase complexity            | Requires computational resources for level evolution |
| **ACCEL + EasyStart**              | Starts with simple levels and gradually increases difficulty | Best overall performance                             |

## Experiments
We tested each method using different hyperparameter configurations:

### Fixed Hyperparameters:
- Parallel Envs: **4**
- Test Levels: **100**
- Batch Size: **128**
- Replay Probability: **0.8**
- Level Buffer Size: **256**
- Initial Fill Size: **128**
- Regret Threshold: **0**
- Train Steps: **grid_size² × n_envs**

### Variable Hyperparameters:
| Grid Size    | Total Iterations | Learning Rate (LR) | Skipped Levels |
|--------------|------------------|--------------------|----------------|
| 6, 8, 10, 12 | grid_size³ (1)   | 1e-4               | Counted        |
| 6, 8, 10, 12 | grid_size² (2)   | 1e-3               | Not counted    |

### Learning Algorithm
- **PPO (Proximal Policy Optimization)**
- **Edit Strategy:** +1 block/replay (ACCEL)

## Key Findings
- **ACCEL and ACCEL-EasyStart outperform other methods**, proving the effectiveness of regret-based level evolution.
- Reducing total iterations and increasing learning rates still yielded promising results.
- **PLR’s regret-based selection improves learning**, but ACCEL maintains complex level progression better.
- **Computational constraints limited further experiments**, suggesting potential improvements with more powerful hardware.


## References
- **[Evolving Curricula with Regret-Based Environment Design](https://arxiv.org/abs/2203.01302)** (ICML 2022) - Jack Parker-Holder, Minqi Jiang, Michael Dennis.
- **[Prioritized Level Replay](https://arxiv.org/abs/2010.03934)** (ICML 2021) - Minqi Jiang, Edward Grefenstette, Tim Rocktäschel.


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
