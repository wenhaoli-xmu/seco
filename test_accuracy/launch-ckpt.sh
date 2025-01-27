torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:12006 \
    --nnodes 1 \
    --nproc_per_node 1 \
    test_accuracy/ckpt.py \
    --env-conf test_accuracy/llama3-8b-ckpt.json \
    --log-step 1
