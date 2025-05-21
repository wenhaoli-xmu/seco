MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    draw_curve/sparse.py \
    --env-conf draw_curve/llama3-8b.json \
    --accum-grad 4 \
    --chunk-size 2048 \
    --log-step 1 \
    --seed 3 \
    --lr 2e-5
