torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:40005 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_curve/spaco.py \
    --env-conf test_curve/llama3-8b.json \
    --accum-grad 4 \
    --chunk-size 128 \
    --chunk-budget 32 \
    --log-step 1 \
    --seed 0 \
    --lr 1e-2