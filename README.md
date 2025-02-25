# Planning to Go Out-of-Distribution
[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](http://www.apache.org/licenses/LICENSE-2.0)

Fork of the [official repository](https://github.com/BayesWatch/ptgood) for [the paper](https://arxiv.org/abs/2310.05723) "Planning to Go Out-of-Distribution in Offline-to-Online Reinforcement learning".

I'm working on modifying the code to use Minari and Gymnasium rather than the deprecated D4RL dependency. Out of an abundance of laziness, I've only supported the Mujoco environments for now.

# Installation
YMV but I've included a requirements.txt file for my environment. I installed both gymnasium[mujoco] and minari[all], as well as torch.

```commandline
pip install -r requirements.txt
```

# Offline pretraining & online fine-tuning
To pretrain offline, set `--offline_steps` to some `int` greater than 0, do not set `--model_file` nor `--rl_file`, and
set `--online_steps` to 0. 

```commandline
python3 main.py \
  --wandb_key wandb.txt \
  --env HalfCheetah-v5 \
  --offline_steps 50000 \
  --online_steps 0 \
  --horizon 5 \
  --r 0.5 \
  --critic_norm \
  --rl_grad_clip 99999 \
  --model_train_freq 1000 \
  --imagination_freq 250 \
  --rl_updates_per 20 \
  --rollout_batch_size 100000 \
  --save_rl_post_online \
  --save_rl_post_offline \
  --custom_filepath None
```

To fine-tune online, set `--offline_steps` to 0, set `--model_file`, `--rl_file`, and
`--ceb_file` to your pretrained checkpoints, and set `--online_steps` to some `int` greater than 0.

```commandline
python3 main.py \
  --wandb_key wandb.txt \
  --env HalfCheetah-v5 \
  --offline_steps 0 \
  --online_steps 51000 \
  --model_file ./models/<model_weights.pt> \
  --rl_file ./policies/<agent_weights.pt> \
  --ceb_file ./ceb_weights/<ceb_weights.pt> \
  --horizon 5 \
  --r 0.5 \
  --critic_norm \
  --rl_grad_clip 99999 \
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
```

# Training encoder with the CEB

```commandline
python3 train_ceb.py \
  --wandb_key wandb.txt \
  --env HalfCheetah-v5 \
  --z_dim 8 \
  --beta 0.01 \
  --bs 512 \
  --n_steps 30000 \
  --model_file ./models/HalfCheetah-v5_<your_model>.pt \
  --rl_file rnd \     
  --ceb_file ./ceb_weights/HalfCheetah-v5
```
Should be using policy, but there was a 🐛 I need to investigate...

# Cite
McInroe, Trevor, et al. "Planning to go out-of-distribution in offline-to-online reinforcement learning." Reinforcement Learning Conference (2024)

