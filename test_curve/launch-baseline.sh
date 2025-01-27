torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:12016 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_curve/baseline.py \
    --env-conf test_curve/llama3-8b-ckpt.json \
    --accum-grad 8 \
    --log-step 1
