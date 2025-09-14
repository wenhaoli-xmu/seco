MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20000))


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_efficiency/test.py \
    --context "[8192, 16384, 32768, 65536, 131072, 262144]" \
    --config test_efficiency/config_blockwise_tp_sparse.json
