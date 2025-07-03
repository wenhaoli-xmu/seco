MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 2 \
    test_efficiency/test.py \
    --context "[10240 * i for i in range(1,100)]" \
    --config test_efficiency/config_blockwise_tp_sparse.json
