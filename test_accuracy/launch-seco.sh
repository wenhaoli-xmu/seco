torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:11103 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_accuracy/seco.py \
    --env-conf test_accuracy/llama3-8b.json \
    --accum-grad 8 \
    --chunk-size 32 \
    --log-step 1
