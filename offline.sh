python3 main.py \
  --wandb_key wandb.txt \
  --env Walker2d-v5 \
  --offline_steps 50000 \
  --online_steps 0 \
  --horizon 5 \
  --model_train_freq 1000 \
  --imagination_freq 1000 \
  --rl_updates_per 20 \
  --rollout_batch_size 100000 \
  --eval_model \
  --save_rl_post_offline 
  # --critic_norm \

