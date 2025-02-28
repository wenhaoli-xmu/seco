torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:12000 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_longbench/ckpt.py \
    --env-conf test_longbench/llama3-8b-inst-tuning-ckpt.json \
    --accum-grad 4 \
    --log-step 1 \
    --seed 0 \
    --save-ckpt ckp/inst-tuning-parallel-0.pth 
