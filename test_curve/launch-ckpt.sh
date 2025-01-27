torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:14000 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_curve/ckpt.py \
    --env-conf test_curve/llama3-8b-ckpt.json \
    --accum-grad 1 \
    --log-step 1
