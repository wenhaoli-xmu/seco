torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:13023 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_latency/spaco.py \
    --env-conf test_latency/llama3-8b.json \
    --accum-grad 8 \
    --chunk-size 32 \
    --chunk-budget 8 \
    --log-step 1
