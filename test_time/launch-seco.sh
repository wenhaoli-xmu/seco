torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:10003 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_time/seco.py \
    --env-conf test_time/llama3-8b.json \
    --accum-grad 8 \
    --chunk-size 32 \
    --log-step 1
