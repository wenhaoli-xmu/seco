ckpt=inst-tuning-parallel-0


rm -r pred/$ckpt
mkdir pred/$ckpt

echo "Running prediction for ${ckpt} ..."
python test_longbench/pred.py --env_conf "test_longbench/llama3-8b-inst-eval.json" --model_max_length 8192 --load-ckpt "ckp/${ckpt}.pth"

echo "Evaluating model for ${ckpt} ..."
python LongBench/eval.py --model $ckpt

echo "Displaying results for ${ckpt} ..."
cat "pred/${ckpt}/result.json"

echo "Finished processing ${ckpt}"
echo "------------------------------------"