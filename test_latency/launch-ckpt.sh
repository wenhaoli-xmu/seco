export HF_ACCESS_TOKEN=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

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_latency/ckpt.py \
    --env-conf test_latency/llama3-8b-ckpt.json \
    --accum-grad 8 \
    --log-step 1 \
    --context "[32768]"
