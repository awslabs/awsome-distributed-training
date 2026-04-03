#!/bin/bash

is_supported_instance() {
    local instance_type="$1"
    local supported_types=(
        "ml.p4d.24xlarge" "ml.p4de.24xlarge" 
        "ml.p5.48xlarge" "ml.p5e.48xlarge" "ml.p5en.48xlarge" "ml.p6e-gb200.36xlarge"
        "ml.trn1.2xlarge" "ml.trn1.32xlarge" "ml.trn1n.32xlarge" "ml.trn2.48xlarge" 
        "ml.trn2u.48xlarge" "ml.p6-b200.48xlarge"
    )
    
    for supported in "${supported_types[@]}"; do
        if [ "$instance_type" = "$supported" ]; then
            return 0
        fi
    done
    return 1
}

validate_instances() {
    echo "Validating instance types..."
    local supported_count=0
    while IFS= read -r node; do
        instance_type=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}')
        if ! is_supported_instance "$instance_type"; then
            echo "Skipping node $node (unsupported instance type: $instance_type)"
        else
            supported_count=$((supported_count + 1))
        fi
    done < <(kubectl get nodes --no-headers -o custom-columns=NAME:.metadata.name)

    if [ "$supported_count" -eq 0 ]; then
        echo "No supported instance types found in the cluster."
        exit 1
    fi
    echo "Found $supported_count supported node(s)."
}

get_supported_nodes() {
    kubectl get nodes --no-headers -o custom-columns=NAME:.metadata.name | while read -r node; do
        instance_type=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}')
        if is_supported_instance "$instance_type"; then
            echo "$node"
        fi
    done
}

get_unique_values() {
    local layer=$1
    for node in $SUPPORTED_NODES; do
        kubectl get node "$node" -o jsonpath="{.metadata.labels.topology\.k8s\.aws/network-node-layer-$layer}"
        echo
    done | sort | uniq
}

validate_instances

SUPPORTED_NODES=$(get_supported_nodes)

echo "Getting layer information..."
layer1=($(get_unique_values 1))
layer2=($(get_unique_values 2))
layer3=($(get_unique_values 3))

# Build Mermaid diagram
mermaid="flowchart TD"
mermaid+=$'\n    A["Cluster Topology"]'

for l1 in "${layer1[@]}"; do
    l1_id=$(echo "$l1" | sed 's/[^a-zA-Z0-9]/_/g')
    mermaid+=$'\n    A --> L1_'"${l1_id}[\"Layer 1: ${l1}\"]"  
done

for l2 in "${layer2[@]}"; do
    l2_id=$(echo "$l2" | sed 's/[^a-zA-Z0-9]/_/g')
    for node in $SUPPORTED_NODES; do
        val=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.topology\.k8s\.aws/network-node-layer-2}')
        if [ "$val" = "$l2" ]; then
            parent=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.topology\.k8s\.aws/network-node-layer-1}')
            break
        fi
    done
    parent_id=$(echo "$parent" | sed 's/[^a-zA-Z0-9]/_/g')
    mermaid+=$'\n    L1_'"${parent_id} --> L2_${l2_id}[\"Layer 2: ${l2}\"]"  
done

for l3 in "${layer3[@]}"; do
    l3_id=$(echo "$l3" | sed 's/[^a-zA-Z0-9]/_/g')
    for node in $SUPPORTED_NODES; do
        val=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.topology\.k8s\.aws/network-node-layer-3}')
        if [ "$val" = "$l3" ]; then
            parent=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.topology\.k8s\.aws/network-node-layer-2}')
            break
        fi
    done
    parent_id=$(echo "$parent" | sed 's/[^a-zA-Z0-9]/_/g')
    mermaid+=$'\n    L2_'"${parent_id} --> L3_${l3_id}[\"Layer 3: ${l3}\"]"  
done

for node in $SUPPORTED_NODES; do
    node_id=$(echo "$node" | sed 's/[^a-zA-Z0-9]/_/g')
    l3_parent=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.topology\.k8s\.aws/network-node-layer-3}')
    l3_parent_id=$(echo "$l3_parent" | sed 's/[^a-zA-Z0-9]/_/g')
    mermaid+=$'\n    L3_'"${l3_parent_id} --> N_${node_id}[\"${node}\"]"  
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
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({startOnLoad:true});</script>
</body></html>
EOF

echo "Topology saved to $OUTPUT_FILE"
open "$OUTPUT_FILE" 2>/dev/null || xdg-open "$OUTPUT_FILE" 2>/dev/null || echo "Open $OUTPUT_FILE in your browser"

