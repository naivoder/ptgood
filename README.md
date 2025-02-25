# Planning to Go Out-of-Distribution
[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](http://www.apache.org/licenses/LICENSE-2.0)

Fork of the [official repository](https://github.com/BayesWatch/ptgood) for [the paper](https://arxiv.org/abs/2310.05723) "Planning to Go Out-of-Distribution in Offline-to-Online Reinforcement learning".

I'm working on modifying the code to use Minari and Gymnasium rather than the deprecated D4RL dependency. Out of an abundance of laziness, I've only supported the Mujoco environments for now.

# Offline pretraining & online fine-tuning
To pretrain offline, set `--offline_steps` to some `int` greater than 0, do not set `--model_file` nor `--rl_file`, and
set `--online_steps` to 0. To fine-tune online, set `--offline_steps` to 0, set `--model_file`, `--rl_file`, and
`--ceb_file` to your pretrained checkpoints, and set `--online_steps` to some `int` greater than 0.

```commandline
python3 main.py \
  --wandb_key wandb.txt \
  --env HalfCheetah-v5 \
  --offline_steps 50000 \
  --online_steps 0 \
  --horizon 5 \
  --r 0.5 \
  --critic_norm \
  --model_train_freq 1000 \
  --imagination_freq 250 \
  --rl_updates_per 20 \
  --rollout_batch_size 100000 \
  --custom_filepath None
```

# Training encoder with the CEB
I haven't touched this code yet. 

```commandline
python3 pretrain_ceb.py --wandb_key ../wandb.txt --env dmc2gym_walker --z_dim 8 --beta 0.01 --bs 512 --n_steps 30000 --custom_filepath ${FILEPATH}/walker_walk_random.json --model_file ./models/dmc2gym_walker_random_31012.pt --rl_file ./policies/dmc2gym_walker_a1-k5_mcn_r0.5-21970-post_offline --critic_norm --horizon 2 --imagination_repeat 3 --ceb_file ./ceb_weights/dmc2gym-walker-random-Bp01-bidir-8
```

# Cite
```
@inproceedings{mcinroe2024ptgood,
    title={Planning to Go Out-of-Distribution in Offline-to-Online Reinforcement Learning},
    author={Trevor McInroe, Adam Jelley, Stefano V. Albrecht, Amos Storkey},
    booktitle={Reinforcement Learning Conference (RLC)}
    year={2024}
}
```
