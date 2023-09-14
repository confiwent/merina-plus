"""
In this file, ppo algorithm is adopted to fine-tune the policy of rate adaptation, gae advantage function and multi-step return are used to calculate the gradients.

!!!! Model-free manner, without likelihood loss for VAE

!!! Add the constraints of optimization problem for action selections (called Action pruning strategy/Masks)

Add the reward normalization, using vmaf quality metric

"""

import os
import numpy as np
import random, datetime, json
import torch
from torch.utils.tensorboard import SummaryWriter
from .ppo_agent_vmaf import PPO_agent
from .replay_memory import ReplayMemory
from .env_wrapper_vmaf import VirtualPlayer

RANDOM_SEED = 42

torch.cuda.set_device(1)
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def ppo_training(
    model_actor_para,
    model_vae_para,
    model_critic_para,
    train_env,
    valid_env,
    args,
    add_str,
    summary_dir,
):
    # Set-up output directories
    dt = datetime.datetime.now().strftime("%y%m%d_%H%M")
    net_desc = "{}_{}".format(dt, "_".join(args.name.split()))
    save_folder = os.path.join(*[summary_dir, net_desc])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    writer = SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, "commandline_args.json")
        with open(params_path, "w") as f:
            json.dump(vars(args), f)
    log_file_name = save_folder + "/log"

    ## set some variables of validation
    mean_value = 53
    max_QoE = {}
    max_QoE[0] = -99999

    # define the parameters of ABR environments
    s_info, s_len, c_len, _, bitrate_versions, _, _, _, _ = train_env.get_env_info()
    br_dim = len(bitrate_versions)
    torch.manual_seed(RANDOM_SEED)

    # define the buffer of trajectories
    memory = ReplayMemory(500 * args.ro_len)

    with open(log_file_name + "_record", "w") as log_file, open(
        log_file_name + "_test_ppo", "w"
    ) as test_log_file:
        # initial the agent and environment
        agent = PPO_agent(args, br_dim, s_info, s_len, c_len, max_QoE)
        agent.initial(model_vae_para, model_actor_para, model_critic_para)
        vp_env = VirtualPlayer(args, train_env, log_file)

        # while True:
        for epoch in range(int(args.epochT)):
            # --------- collect trajectories ---------
            agent.model_eval()
            # vp_env.reset_reward()
            agent.collect_steps(args, memory, vp_env)

            # --------- training the models ----------
            if epoch == args.vap_e and (args.vap or not args.from_il):
                # finish the value function approximation
                agent.finish_approx(summary_dir, add_str)
            (
                vae_kld_loss_,
                policy_loss_,
                value_loss_,
                entropy_loss_,
                policy_mi_loss_,
            ) = agent.train(memory)
            if epoch % int(200) == 0 and epoch > 0:
                agent.annealing()
            writer.add_scalar("Avg_VAE_kld_loss", np.mean(vae_kld_loss_), epoch)
            writer.add_scalar("Avg_Policy_loss", np.mean(policy_loss_), epoch)
            writer.add_scalar("Avg_Value_loss", np.mean(value_loss_), epoch)
            writer.add_scalar("Avg_Entropy_loss", np.mean(entropy_loss_), epoch)
            writer.add_scalar("Avg_MI_loss", np.mean(policy_mi_loss_), epoch)

            if epoch % int(args.valid_i) == 0:
                if epoch >= args.vap_e - 1 or not args.vap:
                    mean_value = agent.valid(
                        args, valid_env, epoch, test_log_file, save_folder, add_str
                    )

            writer.add_scalar("Avg_Return", mean_value, epoch)
            writer.flush()

            memory.clear()
        writer.close()
