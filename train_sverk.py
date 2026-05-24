import os
import time
import numpy as np
import torch
from datetime import datetime
from ppo import PPO
from gym_pybullet_drones.envs.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from gym_pybullet_drones.utils.enums import DroneModel

def evaluate_policy(agent, env, deterministic=True, seed=42):
    """
    Evaluates the policy on one episode and returns the reward.
    """
    obs, info = env.reset(seed=seed)
    ep_reward = 0.0
    done = False
    while not done:
        if deterministic:
            action = agent.get_deterministic_action(obs)
        else:
            action = agent.select_action(obs)
        action = np.expand_dims(action, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated or truncated
    return ep_reward

def train():
    # Parameters
    DEFAULT_OBS = ObservationType.KIN
    DEFAULT_ACT = ActionType.THRUST
    DEFAULT_GUI = False
    CTRL_FREQ = 30
    EPISODE_LEN_SEC = 10

    # PPO hyperparameters
    state_dim = 36
    action_dim = 4
    action_std = 0.6
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = int(2.5e5)
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001
    max_training_timesteps = int(3e6)

    # Evaluation settings
    eval_freq = 10000          # evaluate every 10k steps
    eval_episodes = 5          # number of episodes to evaluate
    best_mean_reward = -float('inf')
    best_model_path = None

    # Creating the environment and the agent
    env = FlyThruGateAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=CTRL_FREQ, drone_model=DroneModel.SVERK_V1)
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # The log directory
    log_dir = "log_dir/sverkTHRUST/"
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(os.path.join(log_dir, "training_log.csv"), "w")
    log_f.write("timestep,episode,mean_reward\n")

    time_step = 0
    i_episode = 0
    update_timestep = EPISODE_LEN_SEC * CTRL_FREQ * 4
    action_std_decay_freq = int(2.5e5)
    print_freq = EPISODE_LEN_SEC * CTRL_FREQ * 10
    log_freq = EPISODE_LEN_SEC * CTRL_FREQ * 2

    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    print("Starting the training...")
    start_time = datetime.now()

    while time_step <= max_training_timesteps:
        obs, info = env.reset()
        current_ep_reward = 0

        for step in range(EPISODE_LEN_SEC * CTRL_FREQ):
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # Обновление PPO
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # Noise attenuation
            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Logging (every log_freq steps)
            if time_step % log_freq == 0:
                log_avg_reward = log_running_reward / max(1, log_running_episodes)
                log_f.write(f"{time_step},{i_episode},{log_avg_reward:.4f}\n")
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            # Printing to the terminal
            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / max(1, print_running_episodes)
                print(f"Episode {i_episode} | Timestep {time_step} | Avg Reward {print_avg_reward:.2f}")
                print_running_reward = 0
                print_running_episodes = 0

            # Evaluating the model every eval_freq steps
            if time_step % eval_freq == 0 and time_step > 0:
                print(f"\nEvaluating the model in step {time_step}...")
                mean_reward = evaluate_policy(ppo_agent, env, deterministic=True, seed=42)
                print(f"Mean reward: {mean_reward:.2f}")
                log_f.write(f"{time_step},EVAL,{mean_reward:.4f}\n")
                log_f.flush()

                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    if best_model_path is not None:
                        os.remove(best_model_path)  # deleting the old best model
                    best_model_path = os.path.join(log_dir, f"best_model_{time_step}.pth")
                    ppo_agent.save(best_model_path)
                    print(f"✅ The new best model is saved: {best_model_path} (reward {mean_reward:.2f})")

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

    # Saving the final model
    final_path = os.path.join(log_dir, "final_model.pth")
    ppo_agent.save(final_path)
    print(f"The final model is saved: {final_path}")

    # Saving hyperparameters
    with open(os.path.join(log_dir, "hyperparams.txt"), "w") as f:
        f.write(f"max_training_timesteps: {max_training_timesteps}\n")
        f.write(f"CTRL_FREQ: {CTRL_FREQ}\n")
        f.write(f"eval_freq: {eval_freq}\n")
        f.write(f"eval_episodes: {eval_episodes}\n")
        f.write(f"best_mean_reward: {best_mean_reward}\n")
        f.write(f"date: {datetime.now()}\n")

    log_f.close()
    env.close()
    print("Training completed.")

if __name__ == "__main__":
    train()
