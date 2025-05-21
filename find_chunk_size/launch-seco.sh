MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    find_chunk_size/seco.py \
    --env-conf find_chunk_size/llama3-8b.json \
    --accum-grad 8 \
    --chunk-size "[512 * (i + 1) for i in range(8)]" \
    --log-step 1 \
    --context 32768 \
    --cpu-offload