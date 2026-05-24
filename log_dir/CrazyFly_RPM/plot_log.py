import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# The path to the log file (can be passed as an argument or specified manually)
if len(sys.argv) > 1:
log_path = sys.argv[1]
else:
log_path = 'PPO_log.csv'

# Check the existence of the file
if not os.path.exists(log_path):
    print(f"The {log_path} file was not found. Specify the correct path.")
sys.exit(1)

# Reading CSV (taking into account possible spaces after commas)
df = pd.read_csv(log_path, skipinitialspace=True)

# Output the first lines for verification
print("The first 5 lines:")
print(df.head())

# Main plot: reward vs timestep
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df['timestep'], df['reward'], alpha=0.5, linewidth=0.5, label='Raw reward')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward progression over timesteps')
plt.grid(True, alpha=0.3)

# Smoothed curve (moving average with a window, for example, 50)
window = 50
if len(df) > window:
    df_smooth = df.rolling(window=window, on='timestep').mean()
    plt.plot(df_smooth['timestep'], df_smooth['reward'], color='red', linewidth=2, label=f'Smoothed (window={window})')
plt.legend()

# Second plot: reward vs episode
plt.subplot(1, 2, 2)
plt.plot(df['episode'], df['reward'], alpha=0.5, linewidth=0.5, label='Raw reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward progression over episodes')
plt.grid(True, alpha=0.3)

# Smoothed curve by episode
if len(df) > window:
    df_smooth_ep = df.rolling(window=window, on='episode').mean()
    plt.plot(df_smooth_ep['episode'], df_smooth_ep['reward'], color='red', linewidth=2, label=f'Smoothed (window={window})')
plt.legend()

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150)
plt.show()

print("Plot saved as training_plot.png")
