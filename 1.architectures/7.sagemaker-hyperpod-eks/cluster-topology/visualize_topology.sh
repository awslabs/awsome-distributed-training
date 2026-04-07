#!/bin/bash
set -euo pipefail

SUPPORTED_TYPES=(
    "ml.p4d.24xlarge" "ml.p4de.24xlarge"
    "ml.p5.48xlarge" "ml.p5e.48xlarge" "ml.p5en.48xlarge" "ml.p6e-gb200.36xlarge"
    "ml.trn1.2xlarge" "ml.trn1.32xlarge" "ml.trn1n.32xlarge" "ml.trn2.48xlarge"
    "ml.trn2u.48xlarge" "ml.p6-b200.48xlarge"
)

is_supported_instance() {
    local instance_type="$1"
    for supported in "${SUPPORTED_TYPES[@]}"; do
        if [ "$instance_type" = "$supported" ]; then
            return 0
        fi
    done
    return 1
}

# Fetch all node data in a single API call and filter to supported nodes
fetch_supported_nodes_json() {
    local all_nodes
    all_nodes=$(kubectl get nodes -o json)

    echo "$all_nodes" | jq -c '[
        .items[] |
        select(
            .metadata.labels["node.kubernetes.io/instance-type"] as $it |
            ['"$(printf '"%s",' "${SUPPORTED_TYPES[@]}" | sed 's/,$//')"'] | index($it)
        ) |
        {
            name: .metadata.name,
            l1: .metadata.labels["topology.k8s.aws/network-node-layer-1"],
            l2: .metadata.labels["topology.k8s.aws/network-node-layer-2"],
            l3: .metadata.labels["topology.k8s.aws/network-node-layer-3"],
            instance_type: .metadata.labels["node.kubernetes.io/instance-type"]
        }
    ]'
}

echo "Fetching cluster node data..."
NODES_JSON=$(fetch_supported_nodes_json)

total_nodes=$(kubectl get nodes --no-headers | wc -l | tr -d ' ')
supported_count=$(echo "$NODES_JSON" | jq 'length')
skipped=$((total_nodes - supported_count))

if [ "$skipped" -gt 0 ]; then
    echo "Skipping $skipped node(s) with unsupported instance types."
fi

if [ "$supported_count" -eq 0 ]; then
    echo "No supported instance types found in the cluster."
    exit 1
fi
echo "Found $supported_count supported node(s)."

echo "Building topology..."

# Extract unique layer values and build diagram from local JSON — no extra API calls
layer1=($(echo "$NODES_JSON" | jq -r '.[].l1 // empty' | sort -u))
layer2=($(echo "$NODES_JSON" | jq -r '.[].l2 // empty' | sort -u))
layer3=($(echo "$NODES_JSON" | jq -r '.[].l3 // empty' | sort -u))

mermaid="flowchart TD"
mermaid+=$'\n    A["Cluster Topology"]'

for l1 in "${layer1[@]}"; do
    l1_id=$(echo "$l1" | sed 's/[^a-zA-Z0-9]/_/g')
    mermaid+=$'\n    A --> L1_'"${l1_id}[\"Layer 1: ${l1}\"]"
done

for l2 in "${layer2[@]}"; do
    l2_id=$(echo "$l2" | sed 's/[^a-zA-Z0-9]/_/g')
    parent=""
    parent=$(echo "$NODES_JSON" | jq -r --arg l2 "$l2" '[.[] | select(.l2 == $l2)][0].l1 // empty')
    parent_id=$(echo "$parent" | sed 's/[^a-zA-Z0-9]/_/g')
    mermaid+=$'\n    L1_'"${parent_id} --> L2_${l2_id}[\"Layer 2: ${l2}\"]"
done

for l3 in "${layer3[@]}"; do
    l3_id=$(echo "$l3" | sed 's/[^a-zA-Z0-9]/_/g')
    parent=""
    parent=$(echo "$NODES_JSON" | jq -r --arg l3 "$l3" '[.[] | select(.l3 == $l3)][0].l2 // empty')
    parent_id=$(echo "$parent" | sed 's/[^a-zA-Z0-9]/_/g')
    mermaid+=$'\n    L2_'"${parent_id} --> L3_${l3_id}[\"Layer 3: ${l3}\"]"
done

echo "$NODES_JSON" | jq -r '.[] | "\(.name) \(.l3)"' | while read -r node l3_parent; do
    node_id=$(echo "$node" | sed 's/[^a-zA-Z0-9]/_/g')
    l3_parent_id=$(echo "$l3_parent" | sed 's/[^a-zA-Z0-9]/_/g')
    mermaid_line="    L3_${l3_parent_id} --> N_${node_id}[\"${node}\"]"
    echo "$mermaid_line"
done | {
    while IFS= read -r line; do
        mermaid+=$'\n'"$line"
    done

    echo "$mermaid"

    # Generate HTML visualization
    OUTPUT_FILE="topology.html"
    cat > "$OUTPUT_FILE" <<EOF
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Cluster Topology</title></head>
<body>
  <pre class="mermaid">
${mermaid}
  </pre>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({startOnLoad:true});</script>
</body></html>
EOF

    echo "Topology saved to $OUTPUT_FILE — open in a browser to view"
}
