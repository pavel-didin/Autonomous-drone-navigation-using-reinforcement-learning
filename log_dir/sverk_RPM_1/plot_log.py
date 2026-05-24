import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if len(sys.argv) > 1:
    log_path = sys.argv[1]
else:
    log_path = 'training_log.csv'

if not os.path.exists(log_path):
    print(f"The {log_path} file was not found. Specify the correct path.")
    sys.exit(1)

df = pd.read_csv(log_path, skipinitialspace=True)

# Convert columns to numeric (coerce errors to NaN)
df['timestep'] = pd.to_numeric(df['timestep'], errors='coerce')
df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
df['mean_reward'] = pd.to_numeric(df['mean_reward'], errors='coerce')

# Drop rows with NaN in any of these columns
df.dropna(subset=['timestep', 'episode', 'mean_reward'], inplace=True)

# Filter timestep (now numeric)
df = df[df['timestep'] <= 2500000]

print("First 5 lines after cleaning:")
print(df.head())

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df['timestep'], df['mean_reward'], alpha=0.5, linewidth=0.5, label='Raw reward')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward progression over timesteps')
plt.grid(True, alpha=0.3)

window = 50
if len(df) > window:
    # Rolling mean works now because all data is numeric
    df_smooth = df.rolling(window=window, on='timestep').mean()
    plt.plot(df_smooth['timestep'], df_smooth['mean_reward'], color='red', linewidth=2, label=f'Smoothed (window={window})')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['episode'], df['mean_reward'], alpha=0.5, linewidth=0.5, label='Raw reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward progression over episodes')
plt.grid(True, alpha=0.3)

if len(df) > window:
    df_smooth_ep = df.rolling(window=window, on='episode').mean()
    plt.plot(df_smooth_ep['episode'], df_smooth_ep['mean_reward'], color='red', linewidth=2, label=f'Smoothed (window={window})')
plt.legend()

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150)
plt.show()

print("Plot saved as training_plot.png")
