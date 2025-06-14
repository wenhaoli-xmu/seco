MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))

hostfile=""

deepspeed \
    --hostfile=$hostfile \
    --launcher SLURM \
    test_efficiency/test_ds.py \
    --deepspeed_config test_efficiency/zero3.json \
    --context "[10240 * i for i in range(1,100)]" \
    --config test_efficiency/config_baseline.json 