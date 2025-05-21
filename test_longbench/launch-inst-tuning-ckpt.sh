export HF_ACCESS_TOKEN=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

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))

deepspeed \
    --launcher SLURM \
    --include localhost:0,1,2,3,4,5,6,7 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --no_ssh_check \
    test_longbench/ckpt.py \
    --deepspeed_config test_longbench/zero2.json \
    --env-conf test_longbench/llama3-8b-inst-tuning-ckpt.json \
    --log-step 1 \
    --seed 0 \
    --save-ckpt ckp/llama3-8b-inst-tuning-ckpt.pth 
