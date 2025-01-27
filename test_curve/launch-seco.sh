torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:14001 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_curve/seco.py \
    --env-conf test_curve/llama3-8b.json \
    --accum-grad 1 \
    --chunk-size 128 \
    --log-step 1
