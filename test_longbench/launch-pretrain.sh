torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:10000 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_longbench/seco.py \
    --env-conf test_longbench/llama3-8b-pretrain.json \
    --accum-grad 4 \
    --chunk-size 256 \
    --log-step 1 \
    --save-ckpt ckp/pretrain.pth
