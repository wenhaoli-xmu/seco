torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:12006 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_latency/ckpt.py \
    --env-conf test_latency/llama3-8b-ckpt.json \
    --accum-grad 8 \
    --log-step 1
