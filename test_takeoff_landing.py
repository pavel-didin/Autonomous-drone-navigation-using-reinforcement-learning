import time
import numpy as np
from ppo import PPO
from gym_pybullet_drones.envs.TakeoffLandingAviary import TakeoffLandingAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel
from gym_pybullet_drones.utils.utils import sync

def test():
    checkpoint_path = "log_dir/takeoff_landing_2/best_model_830000.pth"

    env = TakeoffLandingAviary(
        drone_model=DroneModel.SVERK_V1,
        gui=True,
        record=False,
        act=ActionType.THRUST,
        target_height=0.5,
        episode_len_sec=20.0,
        ctrl_freq=30,
        obs=ObservationType.KIN
    )

    state_dim = 20
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

    max_steps = int((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ)
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
    print(f"Final phase: {info.get('phase')}")
    print(f"Final drone position: {env.pos[0]}")
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    test()
