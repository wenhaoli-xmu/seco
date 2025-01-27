torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:12801 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_curve/spaco.py \
    --env-conf test_curve/llama3-8b.json \
    --accum-grad 1 \
    --chunk-size 128 \
    --chunk-budget 8 \
    --log-step 1
