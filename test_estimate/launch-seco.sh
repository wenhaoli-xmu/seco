torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:14001 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_estimate/seco.py \
    --env-conf test_estimate/llama3-8b.json \
    --accum-grad 1 \
    --chunk-size 16 \
    --sample 8 \
    --log-step 1
