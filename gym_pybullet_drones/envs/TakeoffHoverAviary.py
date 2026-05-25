import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TakeoffHoverAviary(BaseRLAviary):
    """Single agent RL: takeoff from ground and hover at target height."""

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
                 target_height: float = 0.4,
                 episode_len_sec: float = 5.0):
        """
        Parameters
        ----------
        target_height : float
            Desired hover height above ground (m).
        episode_len_sec : float
            Maximum episode duration (seconds).
        """
        self.TARGET_HEIGHT = target_height
        self.EPISODE_LEN_SEC = episode_len_sec

        # Start just above ground to avoid penetration
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

        # For reward computation
        self.prev_dist = None
        self.goal_reached = False

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Reset distance to target
        pos = self._getDroneStateVector(0)[0:3]
        self.prev_dist = np.linalg.norm(np.array([0, 0, self.TARGET_HEIGHT]) - pos)
        self.goal_reached = False
        return obs, info

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]          # roll, pitch, yaw
        ang_vel = state[13:16]     # p, q, r

        target = np.array([0.0, 0.0, self.TARGET_HEIGHT])
        dist = np.linalg.norm(target - pos)

        # Distance change (positive when moving towards target)
        delta_d = self.prev_dist - dist
        self.prev_dist = dist

        # Check if target reached (within 5 cm)
        if dist < 0.05 and not self.goal_reached:
            self.goal_reached = True
            # Optional: give a large sparse bonus
            return 100.0

        # Dense reward components
        penalty_ang = 0.001 * np.linalg.norm(ang_vel)          # discourage fast rotation
        penalty_att = 0.01 * (rpy[0]**2 + rpy[1]**2)          # discourage tilt
        survival_bonus = 0.05                                 # per step alive

        reward = delta_d - penalty_ang - penalty_att + survival_bonus
        # Clip to reasonable range (optional)
        reward = np.clip(reward, -1.0, 1.0)
        return reward

    def _computeTerminated(self):
        """Episode ends successfully when goal is reached and stability condition holds."""
        # Simple termination: when target reached and drone stays there (optional)
        # For now, just let the episode continue until truncated.
        return False

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:9]   # roll, pitch only

        # Out of bounds
        if (abs(pos[0]) > 1.5 or abs(pos[1]) > 1.5 or pos[2] < 0.0 or pos[2] > 1.0):
            return True
        # Excessive tilt (> 30° ≈ 0.52 rad)
        if (abs(rpy[0]) > 0.52 or abs(rpy[1]) > 0.52):
            return True
        # Time limit
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"target_height": self.TARGET_HEIGHT,
                "goal_reached": self.goal_reached}
