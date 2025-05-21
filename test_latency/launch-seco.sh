MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_latency/seco.py \
    --env-conf test_latency/llama3-8b.json \
    --accum-grad 8 \
    --chunk-size 512 \
    --log-step 1 \
    --context "[10240 * i for i in range(1,10)]"
