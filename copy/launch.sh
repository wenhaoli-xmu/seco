export CUDA_VISIBLE_DEVICES=0

python copy/test.py \
    --env_conf copy/vicuna-7b-v1.5.json \
    --chat_template vicuna_v1.1 \
    --enable_prune