# TensorBoard Logging Setup for SAC Training

This implementation automatically creates timestamped directories for each training run, stored in `/home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC/runs/`.

## Features

- **Automatic timestamped directories**: Each run creates a new folder with format `YYYYMMDD_HHMMSS`
- **Organized storage**: All run logs are stored in the centralized `runs/` directory
- **Easy TensorBoard access**: Use the provided script for quick TensorBoard launches

## Usage

### Running Training
When you run `train_SAC.py` or `test.py`, TensorBoard logging is automatically set up with a timestamped directory:

```bash
cd /home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC
python train_SAC.py
```

This will create a directory like `/home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC/runs/20241119_143022/` and log all training metrics there.

### Viewing Results with TensorBoard

A convenient script is provided to launch TensorBoard:

```bash
# Show available runs and usage help
./run_tensorboard.sh

# View all runs simultaneously
./run_tensorboard.sh --all

# View the most recent run only
./run_tensorboard.sh --latest

# View a specific run
./run_tensorboard.sh 20241119_143022
```

TensorBoard will be accessible at `http://localhost:6006`

## Logged Metrics

The following metrics are automatically logged during training:

### Reward Metrics
- `Reward/Train` - Training reward per episode
- `eval/reward` - Evaluation reward (average over test episodes)
- `eval/reward_max` - Maximum evaluation reward
- `eval/reward_min` - Minimum evaluation reward

### UAV Agent Metrics
- `train_uav/actor_loss` - Actor network training loss
- `train_uav/critic1_loss` - Critic 1 network training loss
- `train_uav/critic2_loss` - Critic 2 network training loss
- `train_uav/alpha_loss` - Temperature parameter training loss
- `train_uav/alpha` - Temperature alpha value

### System Metrics
- `Metrics/Sensing_Fairness` - Fairness in sensing operations
- `Metrics/Charging_Fairness` - Fairness in charging operations
- `Metrics/Avg_GN_Energy` - Average ground node energy
- `Metrics/Total_UAV_Energy` - Total UAV energy consumption
- `Metrics/Total_GN_Charge_mJ` - Total ground node charging

### Performance Metrics
- `Performance/Episode_Time_Seconds` - Time per episode
- `Performance/Average_Episode_Time` - Running average episode time
- `Performance/Total_Training_Time_Minutes` - Total training time
- `Performance/GPU_Memory_Used_MB` - GPU memory usage
- `Performance/GPU_Memory_Reserved_MB` - Reserved GPU memory

## Directory Structure

```
/home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC/runs/
├── 20241119_130000/      # First training run (timestamp)
├── 20241119_140000/      # Second training run (timestamp)
├── 20241119_150000/      # Third training run (timestamp)
└── ...
```

## Implementation Details

The tensorboard configuration is implemented in:
- `/home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC/train_SAC.py` - Main training script
- `/home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC/test.py` - Testing script
- `/home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC/run_tensorboard.sh` - TensorBoard launcher script

The logging implementation automatically:
1. Creates the base directory if it doesn't exist
2. Creates a timestamped subdirectory for each run
3. Sets up TensorBoard logging with proper flush intervals
4. Logs comprehensive training metrics during execution