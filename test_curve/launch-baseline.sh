torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:12016 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_curve/baseline.py \
    --env-conf test_curve/llama3-8b.json \
    --accum-grad 1 \
    --log-step 1 \
    --seed 0 \
    --lr 1e-5
