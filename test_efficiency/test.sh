MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20000))


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 4 \
    test_efficiency/test.py \
    --context "[32768, 65536, 131072, 262144, 524288, 1048576]" \
    --config test_efficiency/config_blockwise_sparse.json
