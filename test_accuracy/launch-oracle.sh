torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:14000 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_accuracy/oracle.py \
    --env-conf test_accuracy/qwen2.5-0.5b.json \
    --accum-grad 1 \
    --chunk-size 64 \
    --sample 8 \
    --log-step 1
