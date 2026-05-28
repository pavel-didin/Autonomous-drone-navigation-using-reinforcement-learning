import time
import numpy as np
from gym_pybullet_drones.envs.TakeoffHoverAviary import TakeoffHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel
from gym_pybullet_drones.utils.utils import sync

env = TakeoffHoverAviary(drone_model=DroneModel.SVERK_V2, act=ActionType.RATES, gui=True)
obs, _ = env.reset()
for _ in range(300):
    # Зависание: нулевые угловые скорости, тяга висения (thrust_norm=0)
    action = np.array([[0.0, 0.0, 0.0, 0.0]])
    obs, reward, terminated, truncated, info = env.step(action)
