python3 main.py \
  --wandb_key wandb.txt \
  --env Walker2d-v5 \
  --offline_steps 0 \
  --online_steps 50000 \
  --horizon 5 \
  --r 0.5 \
  --model_train_freq 1000 \
  --imagination_freq 1000 \
  --rl_updates_per 20 \
  --rollout_batch_size 100000 \
  --model_file ./models/Walker2d-v5.pt \
  --rl_file ./policies/Walker2d-v5 \
  --ceb_planner \
  --ceb_planner_noise 0.15 \
  --ceb_width 10 \
  --ceb_depth 5 \
  --act_ceb_pct 1.1 \
  --ceb_beta 0.01 \
  --ceb_z_dim 8 \
  --learned_marginal \
  --ceb_file ./ceb_weights/Walker2d-v5 

  # --ceb_update_freq 10000000 \
  # --critic_norm \
