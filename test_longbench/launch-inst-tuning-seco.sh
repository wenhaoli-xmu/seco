torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:11103 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_longbench/seco.py \
    --env-conf test_longbench/llama3-8b-inst-tuning.json \
    --accum-grad 4 \
    --chunk-size 128 \
    --log-step 1 \
    --seed 3 \
    --save-ckpt ckp/inst-tuning-seco-3.pth 
