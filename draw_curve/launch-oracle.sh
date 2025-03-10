torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:14000 \
    --nnodes 1 \
    --nproc_per_node 1 \
    draw_curve/oracle.py \
    --env-conf draw_curve/llama3-8b.json \
    --accum-grad 4 \
    --log-step 1 \
    --seed 0 \
    --lr 1e-3
