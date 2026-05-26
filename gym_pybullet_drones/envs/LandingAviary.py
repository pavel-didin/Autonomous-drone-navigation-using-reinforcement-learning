import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class LandingAviary(BaseRLAviary):
    """Single agent RL: descend from start_height to target_height."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.SVERK_V1,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.THRUST,
                 start_height: float = 0.5,
                 target_height: float = 0.05,
                 episode_len_sec: float = 6.0):
        self.START_HEIGHT = start_height
        self.TARGET_HEIGHT = target_height
        self.EPISODE_LEN_SEC = episode_len_sec

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, start_height]])
        if initial_rpys is None:
            initial_rpys = np.zeros((1, 3))

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        self.prev_dist = None
        self.goal_reached = False

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        pos = self._getDroneStateVector(0)[0:3]
        target = np.array([0.0, 0.0, self.TARGET_HEIGHT])
        self.prev_dist = np.linalg.norm(target - pos)
        self.goal_reached = False
        return obs, info

    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo,
                                     lo, lo, lo, lo, lo, lo, lo, lo]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi,
                                     hi, hi, hi, hi, hi, hi, hi, hi]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        return self._getDroneStateVector(0).astype('float32')

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        ang_vel = state[13:16]

        target = np.array([0.0, 0.0, self.TARGET_HEIGHT])
        dist = np.linalg.norm(target - pos)
        delta_d = self.prev_dist - dist
        self.prev_dist = dist

        # Penalties
        penalty_ang = 0.001 * np.linalg.norm(ang_vel)
        penalty_att = 0.01 * (rpy[0]**2 + rpy[1]**2)

        reward = delta_d - penalty_ang - penalty_att
        reward = np.clip(reward, -10.0, 10.0)
        return float(reward)

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:9]

        if (abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5 or pos[2] < 0.0 or pos[2] > 1.0):
            return True
        if (abs(rpy[0]) > 0.52 or abs(rpy[1]) > 0.52):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"target_height": self.TARGET_HEIGHT,
                "goal_reached": self.goal_reached}
