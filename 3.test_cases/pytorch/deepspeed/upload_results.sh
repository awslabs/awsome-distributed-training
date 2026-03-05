#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# upload_results.sh - Upload benchmark results to S3 and CloudWatch
#
# Usage:
#   bash upload_results.sh [--results-dir sweep_results] [--region us-east-1]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${1:---results-dir}"
CW_REGION="us-east-1"
S3_REGION="us-west-2"
S3_BUCKET="paragao-new-nemo-squash-container"
CW_NAMESPACE="DeepSpeed/B200Benchmarks"
CW_DASHBOARD_NAME="DeepSpeed-B200-Benchmarks"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --region) CW_REGION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

: "${RESULTS_DIR:=${SCRIPT_DIR}/sweep_results}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

# ============================================================
# 1. Upload JSON files to S3
# ============================================================
echo "=== Uploading results to S3 ==="

# Determine S3 path: benchmark-results/b200/2026/March/04/
YEAR=$(date -u +%Y)
MONTH=$(date -u +%B)
DAY=$(date -u +%d)
S3_PREFIX="benchmark-results/b200/${YEAR}/${MONTH}/${DAY}"

for json_file in "${RESULTS_DIR}"/training_bench_*.json; do
    if [ ! -f "${json_file}" ]; then
        echo "No result JSON files found in ${RESULTS_DIR}"
        break
    fi
    filename=$(basename "${json_file}")
    echo "  Uploading ${filename} -> s3://${S3_BUCKET}/${S3_PREFIX}/${filename}"
    aws s3 cp "${json_file}" "s3://${S3_BUCKET}/${S3_PREFIX}/${filename}" \
        --region "${S3_REGION}" \
        --content-type "application/json"
done

echo "S3 upload complete: s3://${S3_BUCKET}/${S3_PREFIX}/"
echo ""

# ============================================================
# 2. Publish metrics to CloudWatch
# ============================================================
echo "=== Publishing metrics to CloudWatch ==="

for json_file in "${RESULTS_DIR}"/training_bench_*.json; do
    if [ ! -f "${json_file}" ]; then
        break
    fi

    filename=$(basename "${json_file}")

    # Extract metadata and summary using python
    read -r config_name tp pp zero_stage precision avg_tflops avg_step_time timestamp < <(
        python3 -c "
import json, sys
with open('${json_file}') as f:
    d = json.load(f)
m = d['metadata']
s = d['summary']
sc = m.get('sweep_config', {})
print(
    sc.get('config_name', 'unknown'),
    sc.get('tp', 8),
    sc.get('pp', 2),
    sc.get('zero_stage', 1),
    m.get('precision', 'bf16'),
    s.get('steady_state_avg_tflops_per_gpu', 0),
    s.get('steady_state_avg_step_time_s', 0),
    m.get('timestamp', '$(date -u +%Y-%m-%dT%H:%M:%SZ)')
)
"
    )

    echo "  Publishing: ${config_name} (TFLOPS=${avg_tflops}, StepTime=${avg_step_time}s)"

    # Publish with full dimensions
    aws cloudwatch put-metric-data \
        --namespace "${CW_NAMESPACE}" \
        --region "${CW_REGION}" \
        --metric-data "[
            {
                \"MetricName\": \"TFLOPSPerGPU\",
                \"Value\": ${avg_tflops},
                \"Unit\": \"Count\",
                \"Timestamp\": \"${timestamp}\",
                \"Dimensions\": [
                    {\"Name\": \"model_size\", \"Value\": \"103b\"},
                    {\"Name\": \"tp\", \"Value\": \"${tp}\"},
                    {\"Name\": \"pp\", \"Value\": \"${pp}\"},
                    {\"Name\": \"zero_stage\", \"Value\": \"${zero_stage}\"},
                    {\"Name\": \"precision\", \"Value\": \"${precision}\"},
                    {\"Name\": \"config_name\", \"Value\": \"${config_name}\"}
                ]
            },
            {
                \"MetricName\": \"StepTimeSeconds\",
                \"Value\": ${avg_step_time},
                \"Unit\": \"Seconds\",
                \"Timestamp\": \"${timestamp}\",
                \"Dimensions\": [
                    {\"Name\": \"model_size\", \"Value\": \"103b\"},
                    {\"Name\": \"tp\", \"Value\": \"${tp}\"},
                    {\"Name\": \"pp\", \"Value\": \"${pp}\"},
                    {\"Name\": \"zero_stage\", \"Value\": \"${zero_stage}\"},
                    {\"Name\": \"precision\", \"Value\": \"${precision}\"},
                    {\"Name\": \"config_name\", \"Value\": \"${config_name}\"}
                ]
            }
        ]"

    # Also publish with just config_name dimension for easy dashboard queries
    aws cloudwatch put-metric-data \
        --namespace "${CW_NAMESPACE}" \
        --region "${CW_REGION}" \
        --metric-data "[
            {
                \"MetricName\": \"TFLOPSPerGPU\",
                \"Value\": ${avg_tflops},
                \"Unit\": \"Count\",
                \"Timestamp\": \"${timestamp}\",
                \"Dimensions\": [
                    {\"Name\": \"config_name\", \"Value\": \"${config_name}\"}
                ]
            },
            {
                \"MetricName\": \"StepTimeSeconds\",
                \"Value\": ${avg_step_time},
                \"Unit\": \"Seconds\",
                \"Timestamp\": \"${timestamp}\",
                \"Dimensions\": [
                    {\"Name\": \"config_name\", \"Value\": \"${config_name}\"}
                ]
            }
        ]"

done

echo "CloudWatch metrics published to namespace: ${CW_NAMESPACE}"
echo ""

# ============================================================
# 3. Create/Update CloudWatch Dashboard
# ============================================================
echo "=== Creating CloudWatch Dashboard ==="

# Build metric entries dynamically from results
TFLOPS_METRICS=""
STEPTIME_METRICS=""
TABLE_METRICS=""

for json_file in "${RESULTS_DIR}"/training_bench_*.json; do
    if [ ! -f "${json_file}" ]; then
        break
    fi

    config_name=$(python3 -c "
import json
with open('${json_file}') as f:
    d = json.load(f)
print(d['metadata'].get('sweep_config', {}).get('config_name', 'unknown'))
")

    TFLOPS_METRICS+="[\"${CW_NAMESPACE}\",\"TFLOPSPerGPU\",\"config_name\",\"${config_name}\",{\"label\":\"${config_name}\"}],"
    STEPTIME_METRICS+="[\"${CW_NAMESPACE}\",\"StepTimeSeconds\",\"config_name\",\"${config_name}\",{\"label\":\"${config_name}\"}],"
    TABLE_METRICS+="[\"${CW_NAMESPACE}\",\"TFLOPSPerGPU\",\"config_name\",\"${config_name}\",{\"label\":\"${config_name} TFLOPS\"}],"
    TABLE_METRICS+="[\"${CW_NAMESPACE}\",\"StepTimeSeconds\",\"config_name\",\"${config_name}\",{\"label\":\"${config_name} StepTime\"}],"
done

# Remove trailing commas
TFLOPS_METRICS="${TFLOPS_METRICS%,}"
STEPTIME_METRICS="${STEPTIME_METRICS%,}"
TABLE_METRICS="${TABLE_METRICS%,}"

DASHBOARD_BODY=$(cat <<DASH
{
  "widgets": [
    {
      "type": "text",
      "x": 0, "y": 0, "width": 24, "height": 1,
      "properties": {
        "markdown": "# DeepSpeed B200 Benchmark Results - GPT 103B\\nCluster: b200-hyperpod | 8 nodes x 8 B200 GPUs | Namespace: \`${CW_NAMESPACE}\`"
      }
    },
    {
      "type": "metric",
      "x": 0, "y": 1, "width": 24, "height": 8,
      "properties": {
        "title": "TFLOPS/GPU Across Sweep Configurations",
        "view": "bar",
        "region": "${CW_REGION}",
        "stat": "Average",
        "period": 86400,
        "yAxis": {"left": {"label": "TFLOPS/GPU", "showUnits": false}},
        "metrics": [${TFLOPS_METRICS}]
      }
    },
    {
      "type": "metric",
      "x": 0, "y": 9, "width": 24, "height": 8,
      "properties": {
        "title": "Step Time Comparison (seconds)",
        "view": "bar",
        "region": "${CW_REGION}",
        "stat": "Average",
        "period": 86400,
        "yAxis": {"left": {"label": "Step Time (s)", "showUnits": false}},
        "metrics": [${STEPTIME_METRICS}]
      }
    },
    {
      "type": "metric",
      "x": 0, "y": 17, "width": 24, "height": 8,
      "properties": {
        "title": "Summary Metrics Table",
        "view": "table",
        "region": "${CW_REGION}",
        "stat": "Average",
        "period": 86400,
        "metrics": [${TABLE_METRICS}]
      }
    }
  ]
}
DASH
)

aws cloudwatch put-dashboard \
    --dashboard-name "${CW_DASHBOARD_NAME}" \
    --region "${CW_REGION}" \
    --dashboard-body "${DASHBOARD_BODY}"

echo "Dashboard created: ${CW_DASHBOARD_NAME}"
echo "URL: https://${CW_REGION}.console.aws.amazon.com/cloudwatch/home?region=${CW_REGION}#dashboards:name=${CW_DASHBOARD_NAME}"
echo ""
echo "=== Upload Complete ==="
echo "S3: s3://${S3_BUCKET}/${S3_PREFIX}/"
echo "CloudWatch: ${CW_NAMESPACE} in ${CW_REGION}"
echo "Dashboard: ${CW_DASHBOARD_NAME}"
