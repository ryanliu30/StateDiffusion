import os
import matplotlib
from matplotlib import pyplot as plt
from cs285 import envs

from cs285.agents.dynamics_model import DynamicsModel
from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer

import os

import gym
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm
import seaborn as sns
import pandas as pd

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse

from cs285.envs import register_envs

register_envs()

matplotlib.use("agg")

def process_batch(batch):
    result = {}
    names = {}
    for name in batch:
        names[f"{name}"] = []
        if batch[name].ndim == 2:
            for i in range(min(5, batch[name].shape[1])):
                result[f"{name}_{i}"] = batch[name][:, i]
                names[f"{name}"].append(f"{name}_{i}")
        else:
            result[name] = batch[name]
            names[f"{name}"].append(name)
    return result, names

def collect_rollout(
    env: gym.Env,
    dyn_model: DynamicsModel,
    sac_agent: SoftActorCritic,
    ob: np.ndarray,
    rollout_len: int = 1,
):
    obs, acs, rewards, next_obs, dones = [], [], [], [], []
    for _ in range(rollout_len):
        # TODO(student): collect a rollout using the learned dynamics models
        # HINT: get actions from `sac_agent` and `next_ob` predictions from `mb_agent`.
        # Average the ensemble predictions directly to get the next observation.
        # Get the reward using `env.get_reward`.
        ac = sac_agent.get_batched_action(observation=ob)
        outputs = [dyn_model.get_dynamics_predictions(i=i, obs=ob, acs=ac) for i in range(dyn_model.ensemble_size)]
        next_ob = np.stack([i[0] for i in outputs]).mean(0)
        rew = np.stack([i[1] for i in outputs]).mean(0)
        done = np.zeros_like(rew)
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        dones.append(done)

        ob = next_ob
    return {
        "observations": np.concatenate(obs, axis = 0),
        "actions": np.concatenate(acs, axis = 0),
        "rewards": np.concatenate(rewards, axis = 0),
        "next_observations": np.concatenate(next_obs, axis = 0),
        "dones": np.concatenate(dones, axis = 0),
    }


def run_training_loop(
    config: dict, logger: Logger, args: argparse.Namespace, sac_config: dict, diffusion_config: dict
):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = env.spec.max_episode_steps

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our MPC implementation only supports continuous action spaces."

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 2

    # initialize agent
    dyn_model = DynamicsModel(
        env,
        **config["agent_kwargs"],
    ).to(ptu.device)

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    # if doing MBPO, initialize SAC and make that our main agent that we use to
    # collect data and evaluate
    sac_agent = SoftActorCritic(
        env.observation_space.shape,
        env.action_space.shape[0],
        **sac_config["agent_kwargs"],
    ).to(ptu.device)
    synthetic_replay_buffer = ReplayBuffer(sac_config["replay_buffer_capacity"])

    # if doing diffusion, include that:
    if diffusion_config is not None:
        diffusion_model = diffusion_config["make_diffusion"](np.prod(env.observation_space.shape), env.action_space.shape[0]).to(ptu.device)
    

    total_envsteps = 0

    for itr in range(config["num_iters"]):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        if itr == 0:
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env=env,
                policy=utils.RandomPolicy(env),
                min_timesteps_per_batch=config["initial_batch_size"],
                max_length=ep_len, 
            )
        else:
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env=env,
                policy=sac_agent,
                min_timesteps_per_batch=config["batch_size"],
                max_length=ep_len, 
            )

        total_envsteps += envsteps_this_batch
        logger.log_scalar(total_envsteps, "total_envsteps", itr)

        # insert newly collected data into replay buffer
        for traj in trajs:
            replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )

        # update agent's statistics with the entire replay buffer
        all_data = replay_buffer.getall()
        dyn_model.update_statistics(
            obs=all_data["observations"],
            acs=all_data["actions"],
            next_obs=all_data["next_observations"],
            rwds=all_data["rewards"]
        )
        if diffusion_config is not None:
            diffusion_model.update_statistics(
                inputs=diffusion_config["join_predictions"](**all_data)
            )

        # train dynamics
        if (sac_config["mbpo_rollout_length"] > 0) or (diffusion_config is not None and (diffusion_config["sample_what"] == "state")):
            print("Training dynamics...")
            for _ in tqdm.trange(
                config["num_agent_train_steps_per_iter"], dynamic_ncols=True
            ):
                for i in range(config["agent_kwargs"]["ensemble_size"]):
                    batch = replay_buffer.sample(config["train_batch_size"])
                    dyn_model.update(
                        i=i,
                        obs=batch["observations"],
                        acs=batch["actions"],
                        next_obs=batch["next_observations"],
                        rwds=batch["rewards"]
                    )

        if diffusion_config is not None:
            print("Training diffusion...")
            steps = diffusion_config["initial_steps"] if itr == 0 else diffusion_config["steps_per_iter"]
            for _ in tqdm.trange(
                steps, dynamic_ncols=True
            ):
                batch = replay_buffer.sample(diffusion_config["batch_size"])
                _ = diffusion_model.update(
                    diffusion_config["join_predictions"](**batch)
                )

        if (sac_config["mbpo_rollout_length"] > 0) or (diffusion_config is not None):
            print("Collecting synthetic data")
            # collect a rollout using the dynamics model
            num_samples = sac_config["synthetic_amplification"] * config["batch_size"] 
            for _ in tqdm.trange(
                round(num_samples / sac_config["synthesis_batch_size"]), dynamic_ncols=True
            ):
                if diffusion_config is not None:
                    if diffusion_config["sample_what"] == "state":
                        rollout = collect_rollout(
                            env,
                            dyn_model,
                            sac_agent,
                            diffusion_config["split_predictions"](
                                ptu.to_numpy(diffusion_model.synthesize(sac_config["synthesis_batch_size"])), 
                                np.prod(env.observation_space.shape), 
                                env.action_space.shape[0]
                            )["observations"],
                            1,
                        )
                    elif diffusion_config["sample_what"] == "all":
                        rollout = diffusion_config["split_predictions"](
                            ptu.to_numpy(diffusion_model.synthesize(sac_config["synthesis_batch_size"])), 
                            np.prod(env.observation_space.shape), 
                            env.action_space.shape[0]
                        )
                else:
                    rollout = collect_rollout(
                        env,
                        dyn_model,
                        sac_agent,
                        replay_buffer.sample(round(sac_config["synthesis_batch_size"] / min(1, sac_config["mbpo_rollout_length"])))["observations"],
                        sac_config["mbpo_rollout_length"],
                    )
                # insert it into the SAC replay buffer only
                synthetic_replay_buffer.batched_insert(
                    observations=rollout["observations"],
                    actions=rollout["actions"],
                    rewards=rollout["rewards"],
                    next_observations=rollout["next_observations"],
                    dones=rollout["dones"],
                )
            print(f"Inserted {num_samples} synthetic sample to the synthetic buffer.")

        # for MBPO: now we need to train the SAC agent
        print("Training SAC agent...")
        for i in tqdm.trange(
            sac_config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            # train SAC
            if (sac_config["mbpo_rollout_length"] > 0) or (diffusion_config is not None):
                num_real = round(sac_config["batch_size"] / (sac_config["synthetic_to_real_ratio"] + 1))
                num_synthetic = sac_config["batch_size"] - num_real
                batch = replay_buffer.sample(num_real)
                synthetic_batch = synthetic_replay_buffer.sample(num_synthetic)
                batch = {key: np.concatenate([batch[key], synthetic_batch[key]], axis = 0) for key in batch}
            else:
                batch = replay_buffer.sample(sac_config["batch_size"])
            sac_agent.update(
                ptu.from_numpy(batch["observations"]),
                ptu.from_numpy(batch["actions"]),
                ptu.from_numpy(batch["rewards"]),
                ptu.from_numpy(batch["next_observations"]),
                ptu.from_numpy(batch["dones"]).bool(),
                i,
            )

        # Run evaluation
        if config["num_eval_trajectories"] == 0:
            continue
        print(f"Evaluating {config['num_eval_trajectories']} rollouts...")
        trajs = utils.sample_n_trajectories(
            eval_env,
            policy=sac_agent,
            ntraj=config["num_eval_trajectories"],
            max_length=ep_len,
        )
        returns = [t["episode_statistics"]["r"] for t in trajs]
        ep_lens = [t["episode_statistics"]["l"] for t in trajs]

        logger.log_scalar(np.mean(returns), "eval_return", itr)
        logger.log_scalar(np.mean(ep_lens), "eval_ep_len", itr)
        print(f"Average eval return: {np.mean(returns)}")

        if len(returns) > 1:
            logger.log_scalar(np.std(returns), "eval/return_std", itr)
            logger.log_scalar(np.max(returns), "eval/return_max", itr)
            logger.log_scalar(np.min(returns), "eval/return_min", itr)
            logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", itr)
            logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", itr)
            logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", itr)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    sac_agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    itr,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )
    if diffusion_config is not None:
        # make pair plots on training end
        variables = ["observations", "next_observations", "actions", "rewards"]
        real_batch, names = process_batch(replay_buffer.sample(1000))
        synthetic_batch, _ = process_batch(synthetic_replay_buffer.sample(1000))
        df = pd.concat([pd.DataFrame({**real_batch, "type": "real"}), 
                        pd.DataFrame({**synthetic_batch, "type": "synthetic"})], ignore_index=True)
        for i in range(3):
            for j in range(i+1, 4):
                g = sns.PairGrid(df, x_vars=names[variables[i]], y_vars=names[variables[j]], hue="type")
                g.map(sns.kdeplot, common_norm=False)
                g.figure.savefig(os.path.join(logger._log_dir, f"{variables[i]}-{variables[j]} plot"))
                plt.close(g.figure)
        for i in range(4):
            g = sns.PairGrid(df, x_vars=names[variables[i]], y_vars=names[variables[i]], hue="type")
            g.map_upper(sns.scatterplot)
            g.map_lower(sns.kdeplot, common_norm=False)
            g.map_diag(sns.kdeplot, common_norm=False)
            g.figure.savefig(os.path.join(logger._log_dir, f"{variables[i]} plot"))
            plt.close(g.figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamics_config_file", type=str, required=True)
    parser.add_argument("--sac_config_file", type=str, required=True)
    parser.add_argument("--diffusion_config_file", type=str, default=None)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)

    args = parser.parse_args()

    

    config = make_config(args.dynamics_config_file)
    sac_config = make_config(args.sac_config_file)

    if args.diffusion_config_file is not None:
        diffusion_config = make_config(args.diffusion_config_file)
    else:
        diffusion_config = None

    config["log_name"] += f"_seed_{args.seed}"
    
    if args.diffusion_config_file:
        config["log_name"] += f"_diffusion_{diffusion_config['sample_what']}"
    else:
        config["log_name"] += f"_mbpo{sac_config['mbpo_rollout_length']}"

    logger = make_logger(config)

    run_training_loop(config, logger, args, sac_config, diffusion_config)


if __name__ == "__main__":
    main()
