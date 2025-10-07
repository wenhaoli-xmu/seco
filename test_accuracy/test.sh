MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20000))

CONTEXT=1048576
BUDGET=256
ROOT_DIR="/path/to/ctx-${CONTEXT}"

mkdir -p $ROOT_DIR

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 4 \
    test_accuracy/test.py \
    --context $CONTEXT \
    --config test_accuracy/config_blockwise.json \
    --root-dir $ROOT_DIR

# torchrun \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#     --nnodes 1 \
#     --nproc_per_node 1 \
#     test_accuracy/test.py \
#     --context $CONTEXT \
#     --config test_accuracy/config_baseline.json \
#     --root-dir $ROOT_DIR

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 4 \
    test_accuracy/test.py \
    --context $CONTEXT \
    --config test_accuracy/config_blockwise_sparse.json \
    --rewrite-config "{\"page_budget\": ${BUDGET}}" \
    --rewrite-name blockwise-tp-sparse-${BUDGET}.pth \
    --root-dir $ROOT_DIR
