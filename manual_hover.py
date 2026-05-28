import time
import numpy as np
from gym_pybullet_drones.envs.TakeoffHoverAviary import TakeoffHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel
from gym_pybullet_drones.utils.utils import sync

def manual_hover():
    # Create environment with GUI
    env = TakeoffHoverAviary(
        drone_model=DroneModel.SVERK_V2,
        gui=True,
        record=False,
        act=ActionType.THRUST,
        target_height=0.5,          # target height for reference only, not used in control
        episode_len_sec=8.0,
        ctrl_freq=30,
        obs=ObservationType.KIN
    )

    obs, info = env.reset()
    start_time = time.time()
    ep_reward = 0

    # Define constant thrust multiplier (1.0 = exact hover thrust per motor)
    # Slightly above 1.0 to overcome ground effect and inertia.
    # Values: 1.05 .. 1.15 typically work.
    hover_multiplier = 1.08   # adjust as needed (1.0 = hover, >1.0 climbs)

    # Constant action for all motors (normalized, will be converted to thrust)
    constant_action = np.array([0.0, 0.0, 0.0, 0.0])   # zero => hover_thrust * 1.0
    # To increase thrust, we can offset: e.g., constant_action = np.array([0.08, 0.08, 0.08, 0.08])
    # But better to rely on hover_multiplier in environment? Actually our environment's
    # _preprocessAction uses: force = hover_thrust * (1 + beta * action)
    # with beta = 0.3. So to get 1.08 * hover_thrust, we need action = (1.08 - 1)/beta = 0.08/0.3 ≈ 0.267.
    # Let's compute directly.
    beta = 0.3   # from BaseRLAviary._preprocessAction for THRUST
    desired_multiplier = 1.08
    action_val = (desired_multiplier - 1.0) / beta
    constant_action = np.full(4, action_val)

    print(f"Using constant action: {constant_action[0]:.3f} (thrust multiplier = {desired_multiplier:.2f})")

    max_steps = int((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ)
    for i in range(max_steps):
        # Repeat the same action each step
        action = constant_action.copy()
        action = np.expand_dims(action, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        # Synchronize to real time for smooth visualization
        sync(i, start_time, env.CTRL_TIMESTEP)

        # Print position every 3 seconds
        if i % (env.CTRL_FREQ * 3) == 0:
            pos = env.pos[0]
            print(f"Time {i/env.CTRL_FREQ:.1f}s, pos z = {pos[2]:.3f}m")

        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break

    print(f"Manual hover finished. Total reward (unused): {ep_reward:.2f}")
    print(f"Final drone position: {env.pos[0]}")
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    manual_hover()
