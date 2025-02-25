torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:10002 \
    --nnodes 1 \
    --nproc_per_node 1 \
    draw_curve/baseline.py \
    --env-conf draw_curve/llama3-8b-ckpt.json \
    --accum-grad 4 \
    --log-step 1 \
    --seed 0 \
    --lr 3e-4
