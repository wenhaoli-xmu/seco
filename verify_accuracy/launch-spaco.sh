torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:20000 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_accuracy/spaco.py \
    --env-conf test_accuracy/llama3-8b.json \
    --accum-grad 1 \
    --chunk-size 16 \
    --chunk-budget 8 \
    --sample 8 \
    --log-step 1
