import torch
import torch.distributions as td
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import argparse
import wandb
from copy import deepcopy
import os
import minari

from utils.replays import ReplayBuffer, OfflineReplay
from networks.ensembles import DynamicsEnsemble
from utils.termination_fns import termination_fns
from rl.sac import SAC
from ml.mixtures import GMM
from ml.ceb import CEB


def gym_to_minari(env_name: str, quality: str = "medium") -> str:
    """
    Convert a Gymnasium environment name (e.g. HalfCheetah-v5)
    to the corresponding Minari dataset name (e.g. mujoco/halfcheetah/medium-v0).
    """
    base = env_name.split("-")[0].lower()
    return f"mujoco/{base}/{quality}-v0"


def convert_minari_to_dict(dataset):
    """
    Converts a MinariDataset (using iterate_episodes) into a dictionary with keys:
      "observations", "actions", "next_observations", "rewards", "terminals"
    where terminations and truncations are combined into a single done flag.
    """
    all_obs, all_actions, all_rewards = [], [], []
    all_terminals, all_next_obs = [], []

    for episode_data in dataset.iterate_episodes():
        obs = np.array(episode_data.observations)
        actions = np.array(episode_data.actions)
        rewards = np.array(episode_data.rewards)
        terminations = np.array(episode_data.terminations)
        truncations = np.array(episode_data.truncations)
        # Combine terminations and truncations into a binary done flag
        dones = np.logical_or(terminations, truncations).astype(np.int32)
        # Compute next observations by shifting (duplicate last observation to maintain length)
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0) if len(obs) > 1 else obs
        all_obs.append(obs)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_terminals.append(dones)
        all_next_obs.append(next_obs)
    return {
        "observations": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "next_observations": np.concatenate(all_next_obs, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0).reshape(-1, 1),
        "terminals": np.concatenate(all_terminals, axis=0).reshape(-1, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Gymnasium environment name")
    parser.add_argument("--a_repeat", type=int, default=1)
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--custom_filepath", default=None)
    parser.add_argument("--wandb_key", required=True)
    parser.add_argument("--bs", type=int, required=True)
    # parser.add_argument("--traj_length", type=int, required=True)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--z_dim", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--model_file", default=None)
    parser.add_argument("--rl_file", default=None)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--imagination_repeat", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=100000)
    parser.add_argument("--model_train_freq", type=int, default=250)
    parser.add_argument("--reward_penalty", default=None)
    parser.add_argument("--reward_penalty_weight", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--critic_norm", action="store_true")
    parser.add_argument("--rl_grad_clip", type=float, default=999999999)
    parser.add_argument("--ceb_file", required=True)
    parser.add_argument("--ceb_pretrained_file", type=str, default=None)
    args = parser.parse_args()

    if args.critic_norm:
        save_str = args.env + "_norm"
        args.ceb_file += "_norm"
    else:
        save_str = args.env

    # Create Gymnasium environments
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    seed = np.random.randint(0, 100000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    device = "cuda"

    # --- Offline Replay via Minari ---
    simple_ds_name = gym_to_minari(args.env, "simple")
    medium_ds_name = gym_to_minari(args.env, "medium")
    expert_ds_name = gym_to_minari(args.env, "expert")
    print(
        f"Loading Minari datasets: {simple_ds_name}, {medium_ds_name}, {expert_ds_name}"
    )
    ds_simple = minari.load_dataset(simple_ds_name, download=True)
    ds_medium = minari.load_dataset(medium_ds_name, download=True)
    ds_expert = minari.load_dataset(expert_ds_name, download=True)

    # Convert each to a dictionary format
    sim_dict = convert_minari_to_dict(ds_simple)
    med_dict = convert_minari_to_dict(ds_medium)
    exp_dict = convert_minari_to_dict(ds_expert)

    # Create OfflineReplay objects
    offline_replay_sim = OfflineReplay(sim_dict, device, args.custom_filepath)
    offline_replay_med = OfflineReplay(med_dict, device)
    offline_replay_exp = OfflineReplay(exp_dict, device)

    env_lower = args.env.lower()
    if "simple" in env_lower:
        selected_offline = offline_replay_sim
    elif "expert" in env_lower:
        selected_offline = offline_replay_exp
    elif "medium" in env_lower:
        selected_offline = offline_replay_med
    else:
        selected_offline = offline_replay_sim  # default

    # --- Training Replay Buffer for Model Imagination ---
    model_retain_epochs = 1
    epoch_length = args.model_train_freq
    rollout_min = args.horizon
    rollout_max = args.horizon
    base_model_buffer_size = int(
        model_retain_epochs
        * args.rollout_batch_size
        * epoch_length
        / args.model_train_freq
    )
    max_model_buffer_size = (
        base_model_buffer_size * rollout_max * args.imagination_repeat
    )
    min_model_buffer_size = base_model_buffer_size * rollout_min
    training_replay = ReplayBuffer(max_model_buffer_size, state_dim, action_dim, device)
    training_replay.max_size = min_model_buffer_size

    # --- Termination Function ---
    termination_fn = termination_fns[args.env.split("-")[0].lower()]
    print(f"Using termination function: {termination_fn}")

    # --- Dynamics Ensemble & RL Agent Setup ---
    if any(
        x in args.env.lower()
        for x in ["humanoid", "pen", "hammer", "door", "relocate", "quadruped"]
    ):
        bs = 1024
        dynamics_hidden = (
            800 if any(x in args.env.lower() for x in ["cmu", "escape"]) else 400
        )
    else:
        bs = 256
        dynamics_hidden = 200

    dynamics_ens = DynamicsEnsemble(
        7,
        state_dim,
        action_dim,
        [dynamics_hidden] * 4,
        "elu",
        False,
        "normal",
        5000,
        True,
        True,
        512,
        0.001,
        10,
        5,
        None,
        False,
        args.reward_penalty,
        args.reward_penalty_weight,
        None,
        None,
        None,
        args.threshold,
        None,
        device,
    )

    if any(
        x in args.env.lower()
        for x in ["humanoid", "ant", "hammer", "door", "relocate", "quadruped"]
    ):
        agent_mlp = (
            [1024, 1024, 1024]
            if any(x in args.env.lower() for x in ["cmu", "escape"])
            else [512, 512, 512]
        )
    else:
        agent_mlp = [256, 256, 256]
    print(f"Agent MLP: {agent_mlp}")

    agent = SAC(
        state_dim,
        action_dim,
        agent_mlp,
        "elu",
        args.critic_norm,
        -20,
        2,
        1e-4,
        3e-4,
        3e-4,
        0.1,
        0.99,
        0.005,
        [-1, 1],
        256,
        2,
        2,
        None,
        device,
        args.rl_grad_clip,
    )

    # print(agent.critic)
    # checkpoint = torch.load(
    #     "./policies/HalfCheetah-v5_a1-k5_mNone_r0.5-98611-post_offline_critic.pt"
    # )
    # print(checkpoint.keys())

    # --- Load Pretrained Dynamics / RL (if provided) ---
    if args.model_file and args.rl_file:
        print("Loading model and RL weights...")
        dynamics_ens.load_state_dict(torch.load(args.model_file))
        if args.rl_file.lower() == "rnd":
            print("Using random policy for RL.")
        else:
            agent.load(args.rl_file)

    # --- Scale Normalization ---
    print("SCALER BEFORE fitting:", dynamics_ens.scaler.mean, dynamics_ens.scaler.std)
    train_batch, _ = selected_offline.random_split(0, selected_offline.size)
    train_inputs, _ = dynamics_ens.preprocess_training_batch(train_batch)
    dynamics_ens.scaler.fit(train_inputs)
    print("SCALER AFTER fitting:", dynamics_ens.scaler.mean, dynamics_ens.scaler.std)
    dynamics_ens.replay = selected_offline

    # --- Imagination Rollout ---
    print("Performing imagination rollouts to fill training replay buffer...")
    for _ in range(args.imagination_repeat):
        dynamics_ens.imagine(
            args.rollout_batch_size,
            args.horizon,
            agent.actor,
            dynamics_ens.replay,
            training_replay,
            termination_fn,
            False,
        )

    # --- Initialize CEB ---
    if not args.large:
        print(
            f"Initializing small CEB (state_dim={state_dim}, action_dim={action_dim})"
        )
        ceb = CEB(
            state_dim,
            action_dim,
            [256, 128, 64],
            args.z_dim,
            "normal",
            args.beta,
            "cuda",
        )
    else:
        print(
            f"Initializing large CEB (state_dim={state_dim}, action_dim={action_dim})"
        )
        ceb = CEB(
            state_dim,
            action_dim,
            [1024, 512, 256, 128],
            args.z_dim,
            "normal",
            args.beta,
            "cuda",
        )
    ceb.scaler = deepcopy(dynamics_ens.scaler)

    with open(args.wandb_key, "r") as f:
        API_KEY = f.read().strip()
    os.environ["WANDB_API_KEY"] = API_KEY
    os.environ["WANDB_DIR"] = "./wandb"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb"
    wandb.init(
        project="oto-mbpo",
        name=f"CEB_{save_str}",
    )

    # --- Train CEB (or load pretrained) ---
    if not args.ceb_pretrained_file:
        training_info = {
            k: 0
            for k in [
                "forward_i_xz_y",
                "log e(z_e|x)",
                "log b(z_e|y)",
                "forward_i_zy",
                "backward_i_xz_y",
                "log e(z_b|x)",
                "log b(z_b|y)",
                "backward_i_zx",
                "std_z",
                "mu_z",
            ]
        }
        step_count = 0
        for step in tqdm(range(args.n_steps), desc="Training CEB"):
            step_hist = ceb.train_step(args.bs, training_replay, scaler=ceb.scaler)
            step_count += 1
            for k in training_info:
                training_info[k] += step_hist.get(k, 0)
            if step % 5000 == 0 and step > 0:
                avg_info = {k: training_info[k] / step_count for k in training_info}
                print(f"Step {step}: {avg_info}")
                training_info = {k: 0 for k in training_info}
                step_count = 0
                # Update global rate using each offline replay and log metrics.
                for quality, replay in zip(
                    ["simple", "medium", "expert"],
                    [offline_replay_sim, offline_replay_med, offline_replay_exp],
                ):
                    ceb.update_global_rate(replay, scaler=ceb.scaler)
                    print(f"{quality} rate: mean={ceb.rate_mean}, std={ceb.rate_std}")
                    step_hist[f"{quality}_rate_std_upper"] = (
                        ceb.rate_mean + ceb.rate_std
                    )
                    step_hist[f"{quality}_rate_std_lower"] = (
                        ceb.rate_mean - ceb.rate_std
                    )
            wandb.log(step_hist)
    else:
        print(f"Loading pretrained CEB weights from {args.ceb_pretrained_file}")
        ceb.load(args.ceb_pretrained_file)

    # --- Train a Marginal over the Code Space ---
    n_dists = 32
    marginal = GMM(n_dists, args.z_dim)
    marginal_opt = torch.optim.Adam(marginal.parameters(), lr=1e-3)
    ceb.marginal_z = marginal

    for i in tqdm(range(50000), desc="Training Marginal"):
        batch = training_replay.sample(args.bs, True)
        s, a, *_ = batch
        sa = torch.cat([s, a], dim=-1)
        sa = ceb.scaler.transform(sa)
        z_dist = ceb.e_zx(sa, moments=False)
        z = z_dist.mean
        m_log_prob = marginal.log_prob(z).sum(-1, keepdim=True)
        loss = -m_log_prob.mean()
        marginal_opt.zero_grad()
        loss.backward()
        marginal_opt.step()
        wandb.log({"m_kl_loss": loss.item()})
        step_hist = {}
        batch = training_replay.sample(args.bs, True)
        s, a, *_ = batch
        sa = torch.cat([s, a], dim=-1)
        step_hist["rate"] = ceb.compute_rate(sa).mean().item()
        for quality, replay in zip(
            ["simple", "medium", "expert"],
            [offline_replay_sim, offline_replay_med, offline_replay_exp],
        ):
            batch = replay.sample(args.bs, True)
            s, a, *_ = batch
            sa = torch.cat([s, a], dim=-1)
            step_hist[f"{quality}_rate"] = ceb.compute_rate(sa).mean().item()
        if i % 5000 == 0 and i > 0:
            for quality, replay in zip(
                ["training", "simple", "medium", "expert"],
                [
                    training_replay,
                    offline_replay_sim,
                    offline_replay_med,
                    offline_replay_exp,
                ],
            ):
                ceb.update_global_rate(replay, scaler=ceb.scaler)
                step_hist[f"{quality}_rate_std_upper"] = ceb.rate_mean + ceb.rate_std
                step_hist[f"{quality}_rate_std_lower"] = ceb.rate_mean - ceb.rate_std
        wandb.log(step_hist)

    ceb.save(save_str)


if __name__ == "__main__":
    main()
