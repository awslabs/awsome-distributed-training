#!/bin/bash
# Launch the H1 Locomotion Interactive Demo (uses pre-trained rsl_rl checkpoint)
# This is the workshop Module 3 demo - click robots, control with arrow keys
echo "Starting H1 Locomotion Interactive Demo..."
echo "Controls: UP=forward, DOWN=stop, LEFT=turn left, RIGHT=turn right, C=camera toggle"
cd ~/environment/IsaacLab
xhost +
docker kill isaac-lab 2>/dev/null
docker run --shm-size=60g --name isaac-lab --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" --rm --network=host \
  -v /home/ubuntu/environment/shared-efs:/workspace/IsaacLab/TrainedModel \
  -e DISPLAY \
  -e "PRIVACY_CONSENT=Y" \
  isaaclab-batch:latest \
  -c "cd /workspace/IsaacLab && /isaac-sim/python.sh scripts/demos/h1_locomotion.py"
