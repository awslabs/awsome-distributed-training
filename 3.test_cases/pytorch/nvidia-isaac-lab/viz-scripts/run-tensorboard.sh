#!/bin/bash
# Launch TensorBoard to view training metrics
echo "Starting TensorBoard on port 6006..."
docker kill isaac-lab-tb 2>/dev/null
docker run --name isaac-lab-tb --entrypoint bash -d --rm --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -v /home/ubuntu/environment/shared-efs:/workspace/IsaacLab/TrainedModel \
  isaaclab-batch:latest \
  -c "/isaac-sim/python.sh -m tensorboard.main --logdir /workspace/IsaacLab/TrainedModel/isaaclab-logs --host 0.0.0.0 --port 6006"
echo "TensorBoard running at http://localhost:6006"
echo "Open Firefox in DCV desktop to view."
