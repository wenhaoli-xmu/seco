MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    train/baseline.py \
    --env-conf train/qwen2.5-1.5b-yarn-baseline.json \
    --accum-grad 1 \
    --log-step 1 \
    --grad-ckpt \
    --seed 0 \
    --lr 5e-5
