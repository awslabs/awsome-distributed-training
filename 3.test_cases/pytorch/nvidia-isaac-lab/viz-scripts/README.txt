=== Isaac Lab Visualization Scripts ===

Shared EFS mounted at: /home/ubuntu/environment/shared-efs/
  - agent_72000.pt          : Pre-trained model (rsl_rl, 72k epochs)
  - isaaclab-logs/          : HyperPod training checkpoints (synced from FSx)

Scripts:
  run-h1-demo.sh           : Interactive H1 demo (pre-trained rsl_rl model)
                             Click robots, arrow keys to control
  run-skrl-play.sh [path]  : Play back skrl checkpoint from HyperPod training
                             Default: uses best_agent.pt
  run-tensorboard.sh       : Launch TensorBoard for training metrics

To use:
  1. Open a terminal in the DCV desktop
  2. cd ~/environment/viz-scripts
  3. ./run-h1-demo.sh        (for pre-trained demo)
  4. ./run-skrl-play.sh      (for HyperPod-trained model)
