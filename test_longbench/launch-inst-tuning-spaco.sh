torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:12003 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_longbench/spaco.py \
    --env-conf test_longbench/llama3-8b-inst-tuning.json \
    --accum-grad 4 \
    --chunk-size 128 \
    --chunk-budget 16 \
    --log-step 1 \
    --seed 3 \
    --save-ckpt ckp/inst-tuning-spaco-3.pth 
