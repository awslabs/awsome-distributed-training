#! /bin/bash
set -e

if [ -z $AWS_BATCH_JOB_NUM_NODES ]; then 
    echo "setting number of nodes (nnodes) to 1"
    NNODES=1
else 
    NNODES=$AWS_BATCH_JOB_NUM_NODES
    echo "number of nodes (nnodes) = ${NNODES}"
fi

if [ -z $AWS_BATCH_JOB_NODE_INDEX ]; then 
    echo "setting node_rank to 0"
    NODE_RANK=0
else 
    NODE_RANK=$AWS_BATCH_JOB_NODE_INDEX
    echo "node_rank = ${NODE_RANK}"
fi

if [ -z $AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS ]; then 
    echo "setting rdvz_endpoint to localhost"
    RDVZ_ENDPOINT="localhost"
else 
    RDVZ_ENDPOINT=$AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS
    echo "rdvz_endpoint = ${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}"
fi

if [ -z $AWS_BATCH_JOB_ID ]; then 
    echo "setting rdvz_id to 123"
    RDVZ_ID="123"
else 
    # truncate last 2 chars from ID (node number)
    SHORT_ID_LENGTH=${#AWS_BATCH_JOB_ID}-2
    RDVZ_ID=${AWS_BATCH_JOB_ID:0:${SHORT_ID_LENGTH}}
    echo "rdvz_id = ${RDVZ_ID}"
fi


if [ -z $PROC_PER_NODE ]; then 
    echo "setting nproc_per_node to 1"
    PROC_PER_NODE=1
fi

if [ -z $TASK ]; then
    echo "setting task to Ant"
    TASK="Ant"
fi

if [ -z $MAX_ITERATIONS ]; then 
    echo "setting max_iterations to 500"
    MAX_ITERATIONS=500
fi

# run torch distributed
/isaac-sim/python.sh -m torch.distributed.run \
    --nproc_per_node=$PROC_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --rdzv_id=$RDVZ_ID --rdzv_backend=c10d \
    --rdzv_endpoint=$RDVZ_ENDPOINT:5555 \
    scripts/reinforcement_learning/skrl/train.py --distributed --task=$TASK --max_iterations=$MAX_ITERATIONS --headless 
