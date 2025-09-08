MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_accuracy/test.py \
    --context 16384 \
    --config test_accuracy/config_blockwise_offload.json


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_accuracy/test.py \
    --context 16384 \
    --config test_accuracy/config_baseline.json


python test_accuracy/compare.py


rm test_accuracy/baseline.pth
rm test_accuracy/blockwise.pth