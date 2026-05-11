# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

apiVersion: v1
kind: Pod
metadata:
  name: isaacsim-webrtc
  namespace: default
  labels:
    app: isaacsim-webrtc
spec:
  nodeSelector:
    node.kubernetes.io/instance-type: ${VIZ_GPU_INSTANCE_TYPE}
  containers:
  - name: isaacsim
    image: ${IMAGE}
    workingDir: /workspace/IsaacLab
    command: ["/bin/bash", "-c"]
    args:
    - |
      # Note: best_agent.pt is the default checkpoint name for the skrl framework.
      # For other frameworks (rsl_rl, rl_games, sb3), adjust the filename pattern.
      CKPT=$(find ${VIZ_FSX_LOG_DIR} -name best_agent.pt -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2)
      echo "Using checkpoint: $CKPT"
      exec /isaac-sim/python.sh scripts/reinforcement_learning/skrl/play.py \
        --task=${TASK} \
        --checkpoint $CKPT \
        --num_envs ${VIZ_NUM_ENVS} \
        --livestream 2
    env:
    - name: ACCEPT_EULA
      value: "Y"
    - name: PRIVACY_CONSENT
      value: "Y"
    ports:
    - containerPort: 49100
      protocol: TCP
      name: signaling
    - containerPort: 48010
      protocol: UDP
      name: media
    resources:
      requests:
        memory: "16Gi"
        cpu: "8"
        nvidia.com/gpu: "1"
      limits:
        memory: "32Gi"
        cpu: "16"
        nvidia.com/gpu: "1"
        ephemeral-storage: "100Gi"
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
    - name: isaac-cache
      mountPath: /isaac-sim/.cache
    - name: isaac-logs
      mountPath: /isaac-sim/.nvidia-omniverse/logs
    - name: fsx
      mountPath: /fsx
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 8Gi
  - name: isaac-cache
    emptyDir:
      sizeLimit: 50Gi
  - name: isaac-logs
    emptyDir:
      sizeLimit: 5Gi
  - name: fsx
    persistentVolumeClaim:
      claimName: ${FSX_PVC_NAME}
