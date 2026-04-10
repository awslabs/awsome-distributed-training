#!/bin/bash
# Nemotron 3 Super — Submit GRPO Smoke Test to RayCluster on Jakarta EKS
#
# Prerequisites:
#   1. RayCluster running: kubectl apply -f grpo-raycluster.yaml
#   2. Model downloaded to FSx: /fsx/nchkumar/nemotron3-super/models/nemotron3-super-hf
#   3. Training data on FSx: /fsx/nchkumar/nemotron3-super/data/{train,val}.jsonl
#
# Usage:
#   ./submit-grpo-smoke-test.sh
#   ./submit-grpo-smoke-test.sh --dry-run
#
# Monitor:
#   kubectl port-forward -n nchkumar svc/raycluster-nemotron-grpo-head-svc 8265:8265
#   # Then open http://localhost:8265 for Ray dashboard

set -euo pipefail

NAMESPACE="${NAMESPACE:-nchkumar}"
HEAD_POD="raycluster-nemotron-grpo-head-28psd"
CONFIG_PATH="/scratch/nemotron3-super/configs/grpo-smoke-test.yaml"
MODEL_PATH="/scratch/nemotron3-super/models/nemotron3-super-hf"
RESULTS_DIR="/scratch/nemotron3-super/results/grpo-smoke-test"
DRY_RUN="${1:-}"

echo "============================================================"
echo "Nemotron 3 Super — GRPO Smoke Test"
echo "============================================================"
echo "  Namespace:   ${NAMESPACE}"
echo "  Head pod:    ${HEAD_POD}"
echo "  Config:      ${CONFIG_PATH}"
echo "  Model:       ${MODEL_PATH}"
echo "  Results:     ${RESULTS_DIR}"
echo "============================================================"

# Check Ray cluster is ready
echo "Checking Ray cluster status..."
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- ray status 2>/dev/null || {
    echo "ERROR: Ray cluster not ready. Deploy with:"
    echo "  kubectl apply -f grpo-raycluster.yaml"
    exit 1
}
echo ""

# Check model exists on FSx
echo "Checking model on FSx..."
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- ls "${MODEL_PATH}/config.json" 2>/dev/null || {
    echo "WARNING: Model not found at ${MODEL_PATH}"
    echo "  Download with: huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --local-dir ${MODEL_PATH}"
    echo ""
}

if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "[DRY RUN] Would submit:"
    echo "  ray job submit --no-wait -- \\"
    echo "    python examples/run_grpo.py \\"
    echo "    --config ${CONFIG_PATH} \\"
    echo "    policy.model_name=${MODEL_PATH} \\"
    echo "    checkpointing.checkpoint_dir=${RESULTS_DIR}/checkpoints \\"
    echo "    logger.log_dir=${RESULTS_DIR}/logs"
    exit 0
fi

# Submit the GRPO job
echo "Submitting GRPO smoke test job..."
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- \
    ray job submit --no-wait -- \
    bash -c "
        cd /opt/nemo-rl && \
        NRL_IGNORE_VERSION_MISMATCH=1 \
        NRL_VLLM_USE_V1=1 \
        VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        python examples/run_grpo.py \
            --config ${CONFIG_PATH} \
            policy.model_name=${MODEL_PATH} \
            checkpointing.checkpoint_dir=${RESULTS_DIR}/checkpoints \
            logger.log_dir=${RESULTS_DIR}/logs
    "

echo ""
echo "Job submitted. Monitor with:"
echo "  kubectl exec -n ${NAMESPACE} ${HEAD_POD} -- ray job list"
echo "  kubectl port-forward -n ${NAMESPACE} svc/raycluster-nemotron-grpo-head-svc 8265:8265"
echo "  # Then open http://localhost:8265"
