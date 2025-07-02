MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    train/seco.py \
    --env-conf train/qwen2.5-1.5b.json \
    --accum-grad 1 \
    --chunk-size 32768 \
    --offload \
    --log-step 1 \
    --seed 0 \
    --lr 2e-5
