#!/bin/bash

# Script to launch TensorBoard with all training runs
BASE_DIR="/home/qiaohongbo_26/sda/graduation/uav_irs-master/SAC/runs"

echo "Available training run directories:"
echo "=================================="
ls -la $BASE_DIR | grep -E '^d' | awk '{print $9}' | while read dir; do
    if [ "$dir" != "." ] && [ "$dir" != ".." ]; then
        echo "  $dir"
    fi
done
echo "=================================="

if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  $0                    # Show all runs"
    echo "  $0 --all              # Launch TensorBoard with all runs"
    echo "  $0 <timestamp>        # Launch TensorBoard with specific run folder"
    echo "  $0 --latest           # Launch TensorBoard with the most recent run"
    exit 0
fi

case "$1" in
    --all)
        echo "Launching TensorBoard with all runs in $BASE_DIR..."
        tensorboard --logdir="$BASE_DIR" --host=0.0.0.0 --port=6006
        ;;
    --latest)
        LATEST=$(ls -t "$BASE_DIR" | head -1)
        if [ -n "$LATEST" ] && [ -d "$BASE_DIR/$LATEST" ]; then
            echo "Launching TensorBoard with latest run: $LATEST"
            tensorboard --logdir="$BASE_DIR/$LATEST" --host=0.0.0.0 --port=6006
        else
            echo "Error: No runs found in $BASE_DIR"
            exit 1
        fi
        ;;
    *)
        if [ -d "$BASE_DIR/$1" ]; then
            echo "Launching TensorBoard with run: $1"
            tensorboard --logdir="$BASE_DIR/$1" --host=0.0.0.0 --port=6006
        else
            echo "Error: Run directory '$1' not found in $BASE_DIR"
            exit 1
        fi
        ;;
esac