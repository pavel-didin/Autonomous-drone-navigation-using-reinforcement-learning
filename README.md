# Drone racing in gym-pybullet-drones using deep reinforcement learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is a **fork** of [phuongboi/drone-racing-using-reinforcement-learning](https://github.com/phuongboi/drone-racing-using-reinforcement-learning) – a project that applied PPO to fly a quadcopter through gates in the [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) environment.  

We have introduced **key modifications** that made it possible for the drone to **consistently pass all four gates** (the original agent passed at most three). Our work adds support for the **Sverk V1 drone model**, implements **thrust‑based control** (essential for sim‑to‑real transfer), and carefully tunes the reward function and training setup.

## 🚀 Key improvements (our contribution)

- ✅ **Thrust control (`ActionType.THRUST`)** instead of RPM – chosen **for sim‑to‑real transferability**. RPM control assumes ideal motors that linearly convert signals to angular velocities – unrealistic on real hardware. Thrust commands (forces in Newtons) are physically meaningful and can be directly mapped to real motor commands via a calibrated mixer. In simulation, learning difficulty is comparable to RPM, but the learned policy is much more likely to transfer to a real drone.
- ✅ **Hover‑power calibration** – the neutral action (0) corresponds to the thrust needed for hovering (`u_hover ≈ 0.0`). This avoids violent initial behaviour and speeds up convergence.
- ✅ **Extended observation space** – from 20 to 36 dimensions, adding the rotation matrix and distances to the next two gates. This provides explicit geometric information.
- ✅ **Modified reward function** – see equation and table below.
- ✅ **Stricter truncation conditions** – out‑of‑bounds (x, y, z) and maximum tilt angle (45°) stop the episode.
- ✅ **Off‑policy algorithms** (TD3, SAC) were tested alongside PPO; PPO with our custom implementation achieved **100% success** on all four gates.

All experiments were performed with the **Sverk V1** drone (mass 0.216 kg, arm 0.0397 m) using the `FlyThruGateAviary` environment.

## 📈 Reward function (implemented in `FlyThruGateAviary._computeReward()`)

Let $R(s_t, a_t)$ be the reward at step $t$:

Вот наша формула:

$$
R(s_t, a_t) = \left\{
\begin{array}{ll}
R_{\text{pass}}, & \text{if gate passed}, \\
R_{\text{coll}}, & \text{if collision}, \\
R_{\text{approach}} + R_{\text{survival}}, & \text{otherwise}.
\end{array}
\right.
$$

Это объясняет...

with the dense terms defined as

$$
R_{\text{approach}} = \Delta d - \alpha \|\boldsymbol{\omega}\| - \beta(\phi^2 + \theta^2), \qquad
R_{\text{survival}} = \varepsilon.
$$

**Parameters and their values:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $R_{\text{pass}}$ | $+50$ | Sparse bonus for passing a gate |
| $R_{\text{coll}}$ | $+10$ (later changed to $0$) | Collision signal (originally positive to encourage exploration) |
| $\Delta d = d_{\text{prev}} - d_{\text{curr}}$ | – | Reduction of Euclidean distance to the next gate centre |
| $\boldsymbol{\omega} = (p,q,r)$ | – | Angular velocity (rad/s) |
| $\alpha$ | $0.001$ | Weight of angular‑velocity penalty |
| $\phi, \theta$ | – | Roll and pitch angles (rad) |
| $\beta$ | $0.01$ | Quadratic penalty coefficient for attitude deviation |
| $\varepsilon$ | $0.1$ | Survival bonus per step (encourages longer episodes) |

Passing condition: drone enters the bounding box of a gate (x between gate pillars, y within gate depth, z between lower and upper bar). Collision detection uses `p.getContactPoints()` between drone and gate URDF.

## 📊 Results

| Algorithm | Action space | Max. gates passed | Training steps | Success rate |
|-----------|--------------|-------------------|----------------|---------------|
| PPO (mine) | Thrust       | 4 (all)           | 1 M          | 100%          |
| PPO (original) | RPM     | 3                 | 3 M            | 0% (4th gate) |
| TD3 (SB3)  | RPM          | 2                 | 6 M            | –             |
| SAC (SB3)  | RPM          | 2                 | 6 M            | –             |

> 📈 Training curves and videos are available in the `log_dir/` folder.

## 🧰 Installation & usage

**1. Clone the repository**  
```bash
git clone https://github.com/pavel-didin/drone-racing-using-reinforcement-learning.git
cd drone-racing-using-reinforcement-learning
```
**2. Set up a Python environment** (Python 3.8+ recommended)

```bash
conda create -n drones python=3.10
conda activate drones
```
**3. Install the modified gym-pybullet-drones locally**

```bash
bash build_project.sh
```
**4. Install additional dependencies** (if not already installed)

```bash
pip install torch pandas matplotlib opencv-python natsort
```
**5. Train a new policy** (PPO with thrust control, Sverk drone)

```bash
python train_sverk.py
```
Logs and models are saved in `log_dir/sverkTHRUST/`.

**6. Test a pretrained model**

```bash
python test_sverk.py
```
(Edit the `checkpoint_path` inside the script to point to your `.pth` file.)

**7. Plot training curves**

```bash
python plot_log.py log_dir/sverkTHRUST/training_log.csv
```
## 📁 Repository structure (relevant parts)

```text
.
├── gym_pybullet_drones/          # Modified environment (thrust control, extended state)
│   ├── envs/
│   │   ├── BaseRLAviary.py       # Added thrust action type
│   │   ├── FlyThruGateAviary.py  # Custom reward + truncation
│   │   └── ...
│   └── assets/
│       └── sverk_v1.urdf         # Drone model
├── ppo.py                        # PPO implementation
├── train_sverk.py                # Training script for Sverk (thrust, PPO)
├── test_sverk.py                 # Evaluation script
├── train_td3.py / train_sac.py   # Off‑policy baselines (Stable‑Baselines3)
├── plot_log.py                   # Visualise training logs
├── frames_to_video.py            # Convert recorded frames to video
└── log_dir/                      # Training logs and models
```

## 📖 References & acknowledgements

- Original environment: `gym-pybullet-drones` (Panerati et al., IROS 2021)

- Baseline RL code: `PPO-PyTorch` by Nikhil Barhate

- Fork base: `phuongboi/drone-racing-using-reinforcement-learning`

- Paper on autonomous racing: `Song et al., Science Robotics 2023 – inspired the gate‑progress formulation`

If you use this work in your research, please cite the original gym‑pybullet‑drones paper and this repository (see CITATION.cff).

## ⚖️ License
MIT License – see LICENSE file.
