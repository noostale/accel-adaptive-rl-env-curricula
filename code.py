# %% [markdown]
# # ACCEL IMPLEMENTATION

# %%
#print("Start")

# %%
import os
import torch
import wandb
import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from collections import deque

from gymnasium.spaces import Box

from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv, Grid

from minigrid.wrappers import ImgObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

os.environ["WANDB_SILENT"] = "true"


device = 'cpu'
print(f"Using device: {device}")

# ====================================================
# 1. Custom MiniGrid Environment that returns only the image
#    for SB3's PPO (which expects a Box space).
# ====================================================
class MyCustomGrid(MiniGridEnv):
    """
    Simple MiniGrid environment that places random wall tiles
    according to a config dict, returning only the 'image' observation.
    """

    def __init__(self, config=None, solvable_only=False, **kwargs):
        if config is None:
            config = {}
        self.config = config
        self.solvable_only = solvable_only

        # Create a random number generator with the custom seed
        self.rng = np.random.default_rng(seed=self.config.get("seed_val"))

        mission_space = MissionSpace(mission_func=lambda: "get to the green goal square")

        super().__init__(
            grid_size=self.config['width'],
            max_steps=self.config['width'] * self.config['height'] * 2, # max_steps is typically 2x the grid size
            see_through_walls=False,
            agent_view_size=5,                      # Size of the agent's view square
            mission_space=mission_space,
            **kwargs
        )

        # Manually define our observation_space as a single Box (the image).
        # By default, MiniGrid's image shape is (view_size, view_size, 3) if using partial obs,
        # or (height, width, 3) if using full-grid observation. We'll do full-grid here:
        # We'll define (self.height, self.width, 3) as the shape.
        # In practice, "image" shape can vary if partial observations are used.
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype=np.uint8
        )

            
    def _gen_grid(self, width, height):
        """
        Generate a new environment layout ensuring solvability if required.
        """
        
        check_stuck = 0
        while True:  # Keep regenerating until a solvable layout is found
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)

            # Place the goal
            goal_pos = self.config.get("goal_pos")
            if goal_pos is None:
                while True:
                    goal_r = self.rng.integers(1, height - 1)
                    goal_c = self.rng.integers(1, width - 1)
                    if self.grid.get(goal_c, goal_r) is None:
                        self.put_obj(Goal(), goal_c, goal_r)
                        self.config["goal_pos"] = (goal_c, goal_r)
                        break
            else:
                self.put_obj(Goal(), goal_pos[0], goal_pos[1])

            # Place the agent
            start_pos = self.config.get("start_pos")
            if start_pos is None:
                while True:
                    start_r = self.rng.integers(1, height - 1)
                    start_c = self.rng.integers(1, width - 1)
                    if self.grid.get(start_c, start_r) is None and (start_c, start_r) != self.config["goal_pos"]:
                        self.agent_pos = (start_c, start_r)
                        self.agent_dir = self.rng.integers(0, 4)
                        self.config["start_pos"] = (start_c, start_r)
                        break
            else:
                self.agent_pos = start_pos
                self.agent_dir = self.rng.integers(0, 4)
                self.config["start_pos"] = start_pos
            
            placed_blocks = 0
            
            # Maximum number of tries to place the blocks
            max_num_tries = 100
            
            # Place random walls using config parameters
            while placed_blocks < self.config["num_blocks"]:
                max_num_tries -= 1
                r = self.rng.integers(1, height - 1)
                c = self.rng.integers(1, width - 1)
                if max_num_tries <= 0:
                    print("Could not place all blocks in the grid.")
                    break
                if self.grid.get(c, r) is None and (c, r) != self.config["start_pos"] and (c, r) != self.config["goal_pos"]:
                    self.put_obj(Wall(), c, r)
                    placed_blocks += 1
                
            

            # Check solvability if required
            if not self.solvable_only or self._is_solvable():
                break
            
            check_stuck += 1
            if check_stuck > 50:
                print("Re-randomizing start and goal positions...")
                self.config.pop("start_pos", None)
                self.config.pop("goal_pos", None)
                self.rng = np.random.default_rng(seed=self.config.get("seed_val") + check_stuck)

        
    def _is_solvable(self):
        """
        Uses Breadth-First Search (BFS) to check if there's a path 
        from the agent's start position to the goal.
        """
        start_pos = self.config["start_pos"]
        goal_pos = self.config["goal_pos"]
        if not start_pos or not goal_pos:
            return False

        queue = deque([start_pos])
        visited = set()
        visited.add(start_pos)

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal_pos:
                return True

            # Possible moves: up, down, left, right
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                cell_obj = self.grid.get(nx, ny)
                if (
                    1 <= nx < self.width - 1 and  # Stay within grid bounds
                    1 <= ny < self.height - 1 and
                    (nx, ny) not in visited and
                    self.grid.get(nx, ny) is None or isinstance(cell_obj, Goal)
                ):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        return False  # No path found

    def reset(self, **kwargs):
        """
        Override reset to ensure we only return the 'image' array
        instead of a dict with 'image' and 'mission'.
        """
        obs, info = super().reset(**kwargs)
        obs = self._convert_obs(obs)
        
        return obs, info

    def step(self, action):
        """
        Same for step: override to convert the dict observation into an image only.
        """
        obs, reward, done, truncated, info = super().step(action)
        obs = self._convert_obs(obs)
        return obs, reward, done, truncated, info

    def _convert_obs(self, original_obs):
        """
        original_obs is typically {'image':..., 'mission':...}.
        We'll just return original_obs['image'] to get a Box(low=0,high=255) shape.
        """
        return original_obs["image"]
        #return np.transpose(original_obs["image"], (2, 0, 1))
    
    def update_config(self, new_config):
        self.config = new_config
        self.reset()



def random_config(grid_size, num_blocks=None):
    max_blocks = int(((grid_size - 1) * (grid_size - 1)) / 2)
    
    if num_blocks is None:
        num_blocks = np.random.randint(1, max_blocks)
    else:
        num_blocks = min(num_blocks, max_blocks)
        
    config = {
        "width": grid_size,
        "height": grid_size,
        "num_blocks": num_blocks,
        "start_pos": None,
        "goal_pos": None,
        "edited": False,
        "seed_val": np.random.randint(0, 999999),
    }
    
    # Set the start and goal positions
    env = MyCustomGrid(config)
    
    # Reset the environment to get the start and goal positions
    env.reset()
    
    # Get the new config from the environment
    config = env.config
        
    return config

def print_level_from_config(config, solvable_only=False):
    #print("Putting up the level from config:", config)
    env = MyCustomGrid(config, render_mode='rgb_array', solvable_only=solvable_only)
    env.reset()
    full_level_image = env.render()  # This should return an RGB image of the full grid

    plt.figure(figsize=(4, 4))
    plt.imshow(full_level_image)
    plt.title("Level Configuration: " + str(config))
    plt.axis("off")
    plt.show()
    
# Modify an existing configuration, adding randomness.
def edit_config(old_config):
    max_blocks = int(((old_config["width"] - 1) * (old_config["height"] - 1)) / 2)
    
    new_config = dict(old_config)
    
    # Randomly change the number of blocks
    new_number_blocks = old_config["num_blocks"] + np.random.choice([-1, 1, 2, 3])
    
    # Ensure the number of blocks is within bounds
    new_config["num_blocks"] = max(1, min(new_number_blocks, max_blocks))    
    
    # Mark the config as edited
    new_config["edited"] = True
    
    return new_config
    
import numpy as np

"""def edit_config(old_config, difficulty_level=1):

    width, height = old_config["width"], old_config["height"]
    total_cells = width * height

    # Define a baseline max number of blocks
    max_blocks = int(0.6 * total_cells)  # Ensure we don't overcrowd (max 60% coverage)
    
    # Calculate the new number of blocks using a logarithmic scale
    base_growth = int(np.log2(total_cells) * difficulty_level)
    
    # Introduce some randomness while keeping it within a reasonable range
    growth_factor = np.random.randint(base_growth // 2, base_growth + 1)
    
    # Compute the new block count
    new_number_blocks = old_config["num_blocks"] + growth_factor
    
    # Ensure it's within the allowed range
    new_config = dict(old_config)
    new_config["num_blocks"] = max(1, min(new_number_blocks, max_blocks))  
    
    # Mark as edited
    new_config["edited"] = True

    return new_config
"""



# ====================================================
# 2. Simple “level buffer” 
# ====================================================
# class to memorize generated levels and score
class LevelBuffer: 
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []  # will store (config_dict, score)

    def add(self, config, score):
        self.data.append((config, score))
        if len(self.data) > self.max_size:
            self.data.sort(key=lambda x: x[1], reverse=True)
            self.data = self.data[: self.max_size]
            #it memorize only the highest score for each level

    def sample_config(self): 
        # Samples a level from the buffer, weighting the probabilities 
        # based on the scores.
        if len(self.data) == 0:
            return None
        scores = [item[1] for item in self.data]
        total = sum(scores)
        if total <= 1e-9:
            # fallback to uniform
            idx = np.random.randint(len(self.data))
            return self.data[idx][0]
        probs = [s / total for s in scores]
        idx = np.random.choice(len(self.data), p=probs)
        return self.data[idx][0]

# ====================================================
# 3. Utility Functions
# ====================================================

# Calculate regret using Generalized Advantage Estimation (GAE) with Stable-Baselines3's PPO model.
# PLR approximates regret using a score function such as the positive value loss.
def calculate_regret_gae(env, model, max_steps, gamma, lam):
    """
    Calculate regret using Generalized Advantage Estimation (GAE)
    with Stable-Baselines3's PPO model.
    """
    obs, _ = env.reset()
    regrets = []
    rewards = []
    dones = []
    values = []

    for t in range(max_steps):
        # Add batch dimension to the observation tensor
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(device)
        
        # Use the model's policy to get the value and action.
        # For actions, model.predict handles single observations well.
        action, _ = model.predict(obs, deterministic=True)
        
        # Compute the value from the policy.
        value_t = model.policy.predict_values(obs_tensor).item()
        values.append(value_t)
        
        # Perform the step in the environment
        obs, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)
        dones.append(done)

        if done or truncated:
            break

    # Add value of the terminal state (0 if done/truncated)
    if done or truncated:
        terminal_value = 0.0
    else:
        terminal_obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(device)
        terminal_value = model.policy.predict_values(terminal_obs_tensor).item()
    values.append(terminal_value)

    # Compute TD-errors and GAE-like regret score
    for t in range(len(rewards)):
        delta_t = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        discounted_error = (gamma * lam) ** t * delta_t
        regrets.append(max(0, discounted_error))

    # Return the maximum positive regret score (or 0 if empty)
    return max(regrets) if regrets else 0.0


def initialize_ppo(env, learning_rate=1e-4):
    return PPO(
        "MlpPolicy",                    # Multi-layer perceptron policy
        env,                            # environment to learn from
        verbose=0,                      # Display training output
        n_steps=256,                    # Number of steps to run for each environment per update
        batch_size=64,                  # Minibatch size for each gradient update
        learning_rate=learning_rate,    # Learning rate for optimizer
        device=device                   # Use GPU if available
    )
    
# Use vectorized environment
def create_vectorized_env(config, n_envs=4, solvable_only=False):
    """
    Create a vectorized environment with n parallel environments.
    """
    return make_vec_env(lambda: MyCustomGrid(config, solvable_only), n_envs=n_envs, vec_env_cls=SubprocVecEnv)



def evalute_models(load_dim = -1, grid_size = 6, n_eval_episodes = 5, num_levels_per_difficulty = 10):
    
    if load_dim > 0:
        # Load the models
        model_dr = PPO.load(f"models/dr_model_{load_dim}x{load_dim}")
        model_plr = PPO.load(f"models/plr_model_{load_dim}x{load_dim}")
        model_accel = PPO.load(f"models/accel_model_{load_dim}x{load_dim}")
        model_accel_easy = PPO.load(f"models/accel_model_easy_{load_dim}x{load_dim}")

    # Inseert the models in a dictionary
    models = {"DR": model_dr, 'PLR': model_plr, 'ACCEL': model_accel, 'ACCEL-EasyStart': model_accel_easy}

    # Generate n levels difficulties with increasing complexity, for each level generate m configs
    difficulties = 3
    num_levels_per_difficulty = num_levels_per_difficulty

    levels = []
    for i in range(difficulties):
        level = []
        for _ in range(num_levels_per_difficulty):
            cfg = random_config(grid_size, num_blocks=grid_size*(i+1))
            print_level_from_config(cfg, solvable_only=True)
            level.append(cfg)
        levels.append(level)

    # Evaluate the model on the generated levels
    results = {}
    for model_name, model in models.items():
        results[model_name] = []
        for i, level in enumerate(levels):
            print(f"Evaluating {num_levels_per_difficulty} levels of difficulty {i + 1} with {grid_size*(i+1)} blocks for model {model_name}...")
            r = []
            for j, cfg in enumerate(level):
                # Create vectorized environment
                env = create_vectorized_env(cfg, n_envs=4, solvable_only=True)
                mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
                r.append(mean_reward)
            results[model_name].append(r)
        print()
        
    # Print mean rewards for each level
    for model_name in models.keys():
        print(f"Model: {model_name}")
        for i, level in enumerate(levels):
            print(f"Level {i + 1} - Complexity {grid_size*(i+1)}: {np.mean(results[model_name][i]):.2f}")
        print()

    # Boxplot of results, a plot for each level complexity comparing models
    plt.figure(figsize=(12, 6))
    for i, level in enumerate(levels):
        plt.subplot(1, difficulties, i + 1)
        plt.boxplot([results[model_name][i] for model_name in models.keys()])
        plt.xticks([1,2,3,4], [model_name for model_name in models.keys()])
        plt.title(f"Level {i + 1} - Complexity {grid_size*(i+1)}")
        plt.ylabel("Mean Reward")
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    plt.savefig("boxplot.png")
    
    
def main_accel(total_iterations, replay_prob, train_steps, level_buffer_size,
               initial_fill_size, grid_size, n_envs, edit_levels, regret_threshold,
               easy_start, domain_randomization):
    
    # Initialize Weights and Biases
    wandb.init(project="accel", config=config)
    
    # Create a level buffer, a personal class to store levels and scores
    level_buffer = LevelBuffer(max_size=level_buffer_size)
    
    # Generate a random configuration {width, height, num_blocks, start_pos, goal_pos}
    dummy_config = random_config(grid_size)
    
    # Create a vectorized environment, so a wrapper for MyCustomGrid that allows interconnection 
    # between gymnasium and stable-baselines3 to train the model in a vectorized way, since we
    # are using DummyVecEnv, it is not true parallelism
    vectorized_env = create_vectorized_env(dummy_config, n_envs=n_envs)

    # Initialize PPO with vectorized environment
    print("Initializing student model PPO...")
    student_model = initialize_ppo(vectorized_env)
    
    # ====================================================
    # Domain Randomization
    # ====================================================
    
    if domain_randomization:
        iteration, skipped = 0, 0
        while iteration < total_iterations + skipped:
            
            if iteration % 30 == 0:
                print(f"\n=== DOMAIN RANDOMIZATION ITERATION {iteration + 1}/{total_iterations + skipped} SKIPPED: {skipped} ===")
            
            iteration += 1
            # generate a random level
            cfg = random_config(grid_size)
            
            # update the vectorized environment with the selected config and train the model
            #for monitor in vectorized_env.envs:
            #    monitor.env.update_config(cfg)
                
            vectorized_env.env_method("update_config", cfg)

            
            student_model.learn(total_timesteps=train_steps)
            
            # compute regret
            regret = calculate_regret_gae(MyCustomGrid(cfg), student_model, max_steps=1000, gamma=0.99, lam=0.95)
            
            # if regret is below threshold, skip
            if regret <= regret_threshold:
                skipped += 1
                continue
    
        return student_model
            
        

    # ====================================================
    # Initial buffer fill
    # ====================================================
    
    print(f"Populating buffer with {initial_fill_size} initial levels with regret > {regret_threshold}...")
    while len(level_buffer.data) < initial_fill_size:
        
        if easy_start:
            cfg = random_config(grid_size, num_blocks=2)
        else:
            cfg = random_config(grid_size)
        
        #for monitor in vectorized_env.envs:
        #    monitor.env.update_config(cfg)
        
        vectorized_env.env_method("update_config", cfg)
        
        student_model.learn(total_timesteps=100)
        
        regret = calculate_regret_gae(MyCustomGrid(cfg), student_model, max_steps=1000, gamma=0.99, lam=0.95)

        # Skip levels with low regret
        if regret < regret_threshold: continue

        level_buffer.add(cfg, regret)

    # ====================================================
    # Main ACCEL loop
    # ====================================================
    
    iteration_regrets = []
    iteration, skipped = 0, 0
    
    print("\nMain training loop...")
    while iteration < total_iterations + skipped:
        
        if iteration % 30 == 0:
            print(f"\n=== ITERATION {iteration + 1}/{total_iterations + skipped} SKIPPED: {skipped} ===")
        
        iteration += 1
        
        # Decide whether to replay or generate a new level
        use_replay = np.random.rand() < replay_prob

        if not use_replay or len(level_buffer.data) == 0:
            # Create a new random level
            cfg = random_config(grid_size)
            #print("Generated new random level:", cfg)
        else:
            # Sample a level from the buffer
            cfg = level_buffer.sample_config()
            #print("Sampled level from buffer:", cfg)
            
        # Update the vectorized environment with the selected config and train the model
        #for monitor in vectorized_env.envs:
        #    monitor.env.update_config(cfg)
        
        vectorized_env.env_method("update_config", cfg)
        
        student_model.learn(total_timesteps=train_steps)
        
        wandb.log({
            "iteration": iteration,
            "regret_score": regret,
            "regret_threshold": regret_threshold,
            "buffer_size": len(level_buffer.data),
            "value_loss": student_model.logger.name_to_value["train/value_loss"],
            "entropy_loss": student_model.logger.name_to_value["train/entropy_loss"],
            "policy_loss": student_model.logger.name_to_value["train/policy_loss"],
        })

        if use_replay and edit_levels:
            # Edit the level and calculate regret
            cfg = edit_config(cfg)
            #print("Edited level to:", cfg)

        regret = calculate_regret_gae(MyCustomGrid(cfg), student_model, max_steps=1000, gamma=0.99, lam=0.95)
        
        if regret <= regret_threshold:
            #print(f"Regret for current level is {regret:.5f} <= threshold {regret_threshold:.5f}. Skipping...")
            skipped += 1
            continue
        
        if iteration % 30 == 0:
            print(f"Regret for current level: {regret}, buffer size: {len(level_buffer.data)}")
        level_buffer.add(cfg, regret)
        iteration_regrets.append(regret)
        
        # Increase the regret threshold slightly
        regret_threshold += 0.0001
        
    
    # Plot and display the progress
    plt.figure(figsize=(8, 4))
    plt.plot(iteration_regrets, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    plt.title("Regret Progress during Training")
    plt.grid(True)
    plt.show()
    
    # Save the figure
    plt.savefig("regret_progress.png")
    
    
    print("\nDone. Final buffer size:", len(level_buffer.data))
    print("Top-5 hardest levels (config, regret):")
    level_buffer.data.sort(key=lambda x: x[1], reverse=True)
    for i, (cfg, sc) in enumerate(level_buffer.data[:5]):
        print(f"{i + 1}. regret={sc:.5f}, config={cfg}")
        #print_level_from_config(cfg)
        
    print("Top-5 easiest levels (config, regret):")
    level_buffer.data.sort(key=lambda x: x[1])
    for i, (cfg, sc) in enumerate(level_buffer.data[:5]):
        print(f"{i + 1}. regret={sc:.5f}, config={cfg}")
        #print_level_from_config(cfg)
    
    wandb.finish()
    
    return student_model

# %% [markdown]
# # TRAIN

# %%
if __name__ == "__main__":
        
    wandb.finish()
        
    config = {
            "grid_size": 10,
            
            "total_iterations": 200,
            "train_steps": 1000,

            "replay_prob": 0.7,            # Probability of replaying a level and editing it vs. generating a new one
            "level_buffer_size": 128,       # Maximum number of levels to store in the buffer
            "initial_fill_size": 64,       # Number of levels to pre-fill the buffer with
            "regret_threshold": 0.00,      # Minimum regret threshold to consider a level for the buffer
            
            "n_envs": 3,                   # Number of parallel environments to use for training
            
            "edit_levels": True,           # Whether to edit levels during training i.e. ACCEL or PLR
            "easy_start": True,            # Whether to fill the buffer with easy levels first i.e. minimum number of blocks
            "domain_randomization": False, # Whether to use domain randomization
    
        }


    config["domain_randomization"] = True
    config["edit_levels"] = False
    config["easy_start"] = False
    print(f"Running Domain Randomization with config: {config}")
    model_dr = main_accel(**config)
    
    # Save the model
    model_dr.save(f"models/dr_model_{config['grid_size']}x{config['grid_size']}")
    
    print("\n\n============================================\n\n")

    config["domain_randomization"] = False
    config["edit_levels"] = False
    config["easy_start"] = False
    print(f"Running PLR with config: {config}")
    model_plr = main_accel(**config)

    # Save the model
    model_plr.save(f"models/plr_model_{config['grid_size']}x{config['grid_size']}")

    print("\n\n============================================\n\n")


    config["domain_randomization"] = False
    config["edit_levels"] = True
    config["easy_start"] = False
    print(f"Running ACCEL with config: {config}")
    model_accel = main_accel(**config)

    # Save the model
    model_accel.save(f"models/accel_model_{config['grid_size']}x{config['grid_size']}")

    print("\n\n============================================\n\n")
    

    config["domain_randomization"] = False
    config["edit_levels"] = True
    config["easy_start"] = True
    print(f"Running ACCEL with easy start with config: {config}")
    model_accel_easy = main_accel(**config)

    # Save the model
    model_accel_easy.save(f"models/accel_model_easy_{config['grid_size']}x{config['grid_size']}")
    
    # Evaluate the models
    evalute_models(load_dim = config["grid_size"], grid_size = config["grid_size"], n_eval_episodes = 5, num_levels_per_difficulty = 10)
