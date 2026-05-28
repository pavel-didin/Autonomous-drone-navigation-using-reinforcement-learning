import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TakeoffLandingAviary(BaseRLAviary):
    """Takeoff (with climbing bonuses) then landing (only distance reward)."""

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
                 target_height: float = 0.5,
                 episode_len_sec: float = 10.0):
        self.TARGET_HEIGHT = target_height
        self.EPISODE_LEN_SEC = episode_len_sec
        self.GROUND_TARGET = np.array([0.0, 0.0, 0.001])

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.000]])
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
        self.phase = 0   # 0 = takeoff, 1 = landing

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.phase = 0
        target = np.array([0.0, 0.0, self.TARGET_HEIGHT])
        pos = self._getDroneStateVector(0)[0:3]
        self.prev_dist = np.linalg.norm(target - pos)
        return obs, info

    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        shape = (1, 20)
        return spaces.Box(low=lo, high=hi, shape=shape, dtype=np.float32)

    def _computeObs(self):
        return self._getDroneStateVector(0).astype('float32')

    def _get_current_target(self):
        if self.phase == 0:
            return np.array([0.0, 0.0, self.TARGET_HEIGHT])
        else:
            return self.GROUND_TARGET

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        rpy = state[7:10]
        ang_vel = state[13:16]

        target = self._get_current_target()
        dist = np.linalg.norm(target - pos)
        delta_d = self.prev_dist - dist
        self.prev_dist = dist

        penalty_ang = 0.001 * np.linalg.norm(ang_vel)
        penalty_att = 0.01 * (rpy[0]**2 + rpy[1]**2)

        if self.phase == 0:   # Takeoff
            survival_bonus = 0.1
            # Weaken the speed bonus when approaching the target
            speed_bonus_factor = max(0.0, min(1.0, dist / 0.2))
            up_bonus = 0.5 * vel[2] * speed_bonus_factor if vel[2] > 0 else 0.0
            height_bonus = 0.2 * (pos[2] - 0.001)
            reward = delta_d - penalty_ang - penalty_att + survival_bonus + up_bonus + height_bonus
            # Penalty for exceeding the target height
            if pos[2] > self.TARGET_HEIGHT:
                reward -= 0.2 * (pos[2] - self.TARGET_HEIGHT)
        else:                 # Landing
            survival_bonus = 0.05
            reward = delta_d - penalty_ang - penalty_att + survival_bonus
            # Lifting penalty (positive vertical velocity)
            if vel[2] > 0:
                reward -= 0.3 * vel[2]
            # Bonus upon landing, scaled by horizontal precision
            if pos[2] < 0.05:
                horizontal_dist = np.linalg.norm(pos[0:2])
                landing_bonus = max(0.0, 100.0 - 200.0 * horizontal_dist)
                reward += landing_bonus

        reward = np.clip(reward, -10.0, 20.0)

        # Phase switching when the target height is reached (5 cm zone)
        if self.phase == 0 and dist < 0.05:
            self.phase = 1
            self.prev_dist = np.linalg.norm(self.GROUND_TARGET - pos)
            reward += 10.0   # Takeoff completion bonus

        return float(reward)

    def _computeTerminated(self):
        # Terminate when landed (phase == 1 and within 5 cm of ground)
        if self.phase == 1:
            state = self._getDroneStateVector(0)
            pos = state[0:3]
            dist_to_ground = np.linalg.norm(self.GROUND_TARGET - pos)
            if dist_to_ground < 0.05:
                return True
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:9]

        if (abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5 or pos[2] < 0.0 or pos[2] > 1.5):
            return True
        if (abs(rpy[0]) > 0.52 or abs(rpy[1]) > 0.52):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"phase": self.phase}
