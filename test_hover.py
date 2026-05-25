import time
import numpy as np
from ppo import PPO
from gym_pybullet_drones.envs.TakeoffHoverAviary import TakeoffHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel
from gym_pybullet_drones.utils.utils import sync

def test():
    # Path to the trained model (update with your actual file)
    checkpoint_path = "log_dir/hover_thrust/best_model_100000.pth"
    env = TakeoffHoverAviary(
        drone_model=DroneModel.SVERK_V1,
        gui=True,
        record=False,
        act=ActionType.THRUST,
        target_height=0.4,
        episode_len_sec=5.0,
        ctrl_freq=30,
        obs=ObservationType.KIN
    )

    state_dim = 36
    action_dim = 4

    ppo_agent = PPO(state_dim, action_dim,
                    lr_actor=0.0003,
                    lr_critic=0.001,
                    gamma=0.99,
                    K_epochs=80,
                    eps_clip=0.2,
                    action_std_init=0.6)

    ppo_agent.load(checkpoint_path)
    print(f"Model loaded: {checkpoint_path}")

    obs, info = env.reset()
    ep_reward = 0
    start = time.time()

    max_steps = int((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ)   # FIXED
    for i in range(max_steps):
        action = ppo_agent.get_deterministic_action(obs)
        action = np.expand_dims(action, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        sync(i, start, env.CTRL_TIMESTEP)

        if terminated or truncated:
            print(f"Episode finished: terminated={terminated}, truncated={truncated}")
            break

    print(f"Total episode reward: {ep_reward:.2f}")
    print(f"Target height reached: {info.get('goal_reached', False)}")
    print(f"Final drone position: {env.pos[0]}")
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    test()
