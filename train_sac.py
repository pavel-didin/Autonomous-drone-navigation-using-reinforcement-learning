import os
import numpy as np
import time
from datetime import datetime
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ---------- Custom callback ----------
class GateSuccessCallback(BaseCallback):
    """The callback that stops learning when all 4 gates are passed."""
    def __init__(self, log_dir, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.success = False
        self.success_timestep = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Getting access to the original environment (the first environment in a vectorized wrapper)
        env = self.model.env.envs[0].env
        if hasattr(env, 'passing_flag') and all(env.passing_flag):
            self.success = True
            self.success_timestep = self.num_timesteps
            print(f"\n🎉 SUCCESS! The drone passed all 4 gates in the {self.success_timestep} step!")
            success_path = os.path.join(self.log_dir, f"SUCCESS_{self.success_timestep}.zip")
            self.model.save(success_path)
            print(f"✅ Successful model saved: {success_path}")
            self.model.stop_training = True

# ---------- Parameters ----------
DEFAULT_OBS = ObservationType.KIN
DEFAULT_ACT = ActionType.RPM
DEFAULT_GUI = False
CTRL_FREQ = 30
TOTAL_TIMESTEPS = 6_000_000

SAC_PARAMS = {
    'learning_rate': 3e-4,
    'buffer_size': 1_000_000,
    'learning_starts': 5000,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'ent_coef': 'auto',
    'target_entropy': 'auto',
    'use_sde': False,
    'sde_sample_freq': -1,
    'policy_kwargs': dict(net_arch=[256, 256])
}

# ---------- Creating a log folder ----------
base_log_dir = "sac_logs"
os.makedirs(base_log_dir, exist_ok=True)
existing_runs = [d for d in os.listdir(base_log_dir) if d.startswith("SAC_") and os.path.isdir(os.path.join(base_log_dir, d))]
run_numbers = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
next_run = max(run_numbers) + 1 if run_numbers else 1
log_dir = os.path.join(base_log_dir, f"SAC_{next_run}")
os.makedirs(log_dir)
print(f"Logs will be saved in: {log_dir}")

# ---------- Diagnosis of the dimension of observations ----------
temp_env = FlyThruGateAvitary(gui=False, obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=CTRL_FREQ, record=False)
obs, info = temp_env.reset()
real_obs_shape = obs.shape
print(f"Real obs shape: {real_obs_shape}")
print(f"observation_space from the temporary environment: {temp_env.observation_space.shape}")
temp_env.close()

# ---------- Creating the main environment ----------
raw_env = FlyThruGateAvitary(gui=DEFAULT_GUI, obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=CTRL_FREQ, record=False)

# Patch _computeTerminated to complete after 4 gates using raw_env
original_terminated = raw_env._computeTerminated
def new_terminated():
    if raw_env.passing_flag[3] or raw_env.collide:
        return True
    return False
raw_env._computeTerminated = new_terminated
print("✅ The episode ends after passing through the 4th gate.")

# Коррекция observation_space
if raw_env.observation_space.shape != real_obs_shape:
    print(f"Incongruity: observation_space={raw_env.observation_space.shape}, real obs={real_obs_shape}. Correcting it...")
    raw_env.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=real_obs_shape, dtype=np.float32)

# Wrap it in a Monitor
env = Monitor(raw_env, log_dir)

# ---------- Creating the SAC model ----------
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    device="cpu",
    **SAC_PARAMS
)

# ---------- Launching training ----------
print("Starting SAC training...")
success_callback = GateSuccessCallback(log_dir)
start_time = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name=f"SAC_run_{next_run}",
    callback=success_callback
)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# ---------- Result processing ----------
if success_callback.success:
    print(f"\n🏆 The training was completed ahead of schedule in step {success_callback.success_timestep}!")
    final_model_path = os.path.join(log_dir, f"SUCCESS_{success_callback.success_timestep}.zip")
else:
    print(f"\n⏱️ The step limit ({TOTAL_TIMESTEPS}) has been reached without passing all the gates.")
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)

print(f"Final model: {final_model_path}")

# ---------- Saving hyperparameters ----------
with open(os.path.join(log_dir, "hyperparams.txt"), "w") as f:
    f.write(f"TOTAL_TIMESTEPS: {TOTAL_TIMESTEPS}\n")
    f.write(f"CTRL_FREQ: {CTRL_FREQ}\n")
    f.write(f"SUCCESS_ACHIEVED: {success_callback.success}\n")
    f.write(f"SUCCESS_TIMESTEP: {success_callback.success_timestep if success_callback.success else 'N/A'}\n")
    for key, value in SAC_PARAMS.items():
        f.write(f"{key}: {value}\n")
    f.write(f"date: {datetime.now()}\n")

print(f"The model is saved in {log_dir}")
