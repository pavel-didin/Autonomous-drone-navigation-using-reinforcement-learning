import os
import time
import numpy as np
from datetime import datetime
from ppo import PPO
from gym_pybullet_drones.envs.TakeoffHoverAviary import TakeoffHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel

def evaluate_policy(agent, env, deterministic=True, eval_episodes=5):
    total_reward = 0
    for _ in range(eval_episodes):
        obs, info = env.reset()
        ep_reward = 0
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
        total_reward += ep_reward
    return total_reward / eval_episodes

def train():
    # Environment
    env = TakeoffHoverAviary(
        drone_model=DroneModel.SVERK_V1,
        gui=False,
        act=ActionType.THRUST,
        target_height=0.4,
        episode_len_sec=5.0,
        ctrl_freq=30
    )

    state_dim = 36   # same extended state as in FlyThruGateAviary
    action_dim = 4   # thrust for 4 motors

    # PPO hyperparameters (similar to train_sverk.py)
    ppo_agent = PPO(state_dim, action_dim,
                    lr_actor=0.0003,
                    lr_critic=0.001,
                    gamma=0.99,
                    K_epochs=80,
                    eps_clip=0.2,
                    action_std_init=0.6)

    log_dir = "log_dir/hover_thrust/"
    os.makedirs(log_dir, exist_ok=True)
    log_f = open(os.path.join(log_dir, "training_log.csv"), "w")
    log_f.write("timestep,episode,mean_reward\n")

    max_timesteps = 1_000_000
    update_timestep = env.EPISODE_LEN_SEC * env.CTRL_FREQ * 4
    eval_freq = 20_000
    log_freq = env.EPISODE_LEN_SEC * env.CTRL_FREQ * 2

    time_step = 0
    episode = 0
    running_reward = 0
    running_episodes = 0
    best_mean_reward = -float('inf')

    print("Starting hover training...")
    start_time = datetime.now()

    while time_step <= max_timesteps:
        obs, info = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            ep_reward += reward

            # PPO update
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # Logging
            if time_step % log_freq == 0 and running_episodes > 0:
                avg_reward = running_reward / running_episodes
                log_f.write(f"{time_step},{episode},{avg_reward:.4f}\n")
                log_f.flush()
                running_reward = 0
                running_episodes = 0

            # Evaluation
            if time_step % eval_freq == 0 and time_step > 0:
                mean_reward = evaluate_policy(ppo_agent, env, deterministic=True)
                print(f"Step {time_step}: eval reward = {mean_reward:.2f}")
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_path = os.path.join(log_dir, f"best_model_{time_step}.pth")
                    ppo_agent.save(best_path)
                    print(f"✅ New best model saved: {best_path}")

        running_reward += ep_reward
        running_episodes += 1
        episode += 1

        if episode % 100 == 0:
            print(f"Episode {episode}, step {time_step}, last reward {ep_reward:.2f}")

    # Final save
    ppo_agent.save(os.path.join(log_dir, "final_model.pth"))
    log_f.close()
    env.close()
    print(f"Training finished. Best reward: {best_mean_reward:.2f}")

if __name__ == "__main__":
    train()
