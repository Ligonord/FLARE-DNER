for seed in 52 869 1001 50542 353778
do
  python train.py --output_dir ../../../../data/Experiments/share14/$seed \
  --train_filepath ../../../../data/Experiments/share14/train.txt \
  --dev_filepath ../../../../data/Experiments/share14/dev.txt \
  --test_filepath ../../../../data/Experiments/share14/test.txt \
  --log_filepath ../../../../data/Experiments/share14/$seed/train.log \
  --model_type elmo --pretrained_model_dir ../../../../data/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
  --weight_decay 0.0001 --max_grad_norm 5 \
  --learning_rate 0.001 --num_train_epochs 20 --patience 5 --eval_metric f1-overall \
  --max_save_checkpoints 0 \
  --cuda_device 0 --seed $seed
done