python3 main.py \
  --wandb_key wandb.txt \
  --env HalfCheetah-v5 \
  --offline_steps 0 \
  --online_steps 51000 \
  --model_file ./models/HalfCheetah-v5_a1_None_step753_69475.pt \
  --rl_file ./policies/HalfCheetah-v5_a1-k5_mNone_r0.5-98611-post_offline \
  --ceb_file ./ceb_weights/HalfCheetah-v5 \
  --horizon 5 \
  --r 0.5 \
  --critic_norm \
  --model_train_freq 1000 \
  --imagination_freq 250 \
  --rl_updates_per 20 \
  --rollout_batch_size 100000 \
  --ceb_planner \
  --ceb_planner_noise 0.3 \
  --ceb_width 10 \
  --ceb_depth 5 \
  --act_ceb_pct 1.1 \
  --ceb_beta 0.01 \
  --ceb_z_dim 8 \
  --ceb_update_freq 10000000 \
  --learned_marginal
