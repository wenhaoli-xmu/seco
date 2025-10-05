MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20000))

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 4 \
    train/train.py \
    --config train/config_blockwise_sparse.json
