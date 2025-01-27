torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:13015 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_accuracy/spaco.py \
    --env-conf test_accuracy/llama3-8b.json \
    --accum-grad 8 \
    --chunk-size 128 \
    --chunk-budget 8 \
    --log-step 1
