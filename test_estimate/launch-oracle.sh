torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:14000 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_estimate/oracle.py \
    --env-conf test_estimate/llama3-8b.json \
    --accum-grad 1 \
    --chunk-size 16 \
    --sample 100 \
    --log-step 1
