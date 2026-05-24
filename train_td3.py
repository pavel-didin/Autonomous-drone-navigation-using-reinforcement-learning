import os
import numpy as np
import time
from datetime import datetime
from gymnasium import spaces

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ---------- Parameters ----------
DEFAULT_OBS = ObservationType.KIN
DEFAULT_ACT = ActionType.RPM
DEFAULT_GUI = False
CTRL_FREQ = 30
TOTAL_TIMESTEPS = 6_000_000

# ---------- Creating a folder of logs with auto-increment ----------
base_log_dir = "td3_logs"
os.makedirs(base_log_dir, exist_ok=True)
existing_runs = [d for d in os.listdir(base_log_dir) if d.startswith("TD3_") and os.path.isdir(os.path.join(base_log_dir, d))]
run_numbers = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
next_run = max(run_numbers) + 1 if run_numbers else 1
log_dir = os.path.join(base_log_dir, f"TD3_{next_run}")
os.makedirs(log_dir)
print(f"Logs will be saved in: {log_dir}")

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,           # save every 100,000 steps
    save_path=log_dir,
    name_prefix="td3_model",
    save_replay_buffer=False,    # Use True if you need to save a buffer.
    save_vecnormalize=False
)

# ---------- Custom callback for checking the passage of all gates ----------
class GateSuccessCallback(BaseCallback):
    """A callback that stops learning when all 4 gates are passed."""
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
            self.model.stop_training = True  # stopping the training

# ---------- Creating the main environment ----------
raw_env = FlyThruGateAviary(gui=DEFAULT_GUI, obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=CTRL_FREQ, record=False)

# Patch _computeTerminated to complete after the 4th gate (using raw_env)
original_terminated = raw_env._computeTerminated
def new_terminated():
    if raw_env.passing_flag[3] or raw_env.collide:
        return True
    return False
raw_env._computeTerminated = new_terminated
print("✅ The episode ends after passing through the 4th gate.")

# Correction of observation_space
if raw_env.observation_space.shape != real_obs_shape:
    print(f"Incongruity: observation_space={raw_env.observation_space.shape}, real obs={real_obs_shape}. Correcting it...")
    raw_env.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=real_obs_shape, dtype=np.float32)

env = Monitor(raw_env, log_dir)

# ---------- TD3 requires specifying the action noiseTD3 требует указания шума действий ----------
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.0 * np.ones(n_actions))

# ---------- Creating a model ----------
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=3e-4,
    buffer_size=200_000,
    learning_starts=20_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=100,
    gradient_steps=100,
    policy_kwargs=dict(net_arch=[400, 300]),
    verbose=1,
    tensorboard_log=log_dir,
    device="cpu"
)

# ---------- Creating a callback ----------
success_callback = GateSuccessCallback(log_dir)

# ---------- Launching training ----------
print("Starting TD3 training...")
start_time = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name=f"TD3_run_{next_run}",
    callback=[success_callback, checkpoint_callback]
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

print(f"The final model: {final_model_path}")

# ---------- Saving hyperparameters ----------
with open(os.path.join(log_dir, "hyperparams.txt"), "w") as f:
    f.write(f"TOTAL_TIMESTEPS: {TOTAL_TIMESTEPS}\n")
    f.write(f"CTRL_FREQ: {CTRL_FREQ}\n")
    f.write(f"SUCCESS_ACHIEVED: {success_callback.success}\n")
    f.write(f"SUCCESS_TIMESTEP: {success_callback.success_timestep if success_callback.success else 'N/A'}\n")
    f.write(f"learning_rate: 1e-3\n")
    f.write(f"buffer_size: 200000\n")
    f.write(f"batch_size: 256\n")
    f.write(f"action_noise_sigma: 0.1\n")
    f.write(f"net_arch: [400,300]\n")
    f.write(f"date: {datetime.now()}\n")

print(f"The model is saved in {log_dir}")
