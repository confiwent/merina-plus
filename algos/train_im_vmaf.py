"""design for IMRL

The expert strategy: MPC oracle

"""

import os
import sys
import numpy as np
import pdb
import json
import datetime
import logging

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import logging
from torch.utils.tensorboard import SummaryWriter

# from algos.mpc_pruning import A_DIM
# # from model_AQ import Actor, Critic
# from .agent_il_v4 import IML_agent
from .AC_net_vmaf import Actor
from .beta_vae_vmaf import BetaVAE

# from .MPC_expert_vmaf_v1 import ABRExpert
from .test_vmaf import valid
from .replay_memory import ReplayMemory
from .helper import save_models_v1

RANDOM_SEED = 28
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_VAE = 1e-4
MAX_GRAD_NORM = 5.0
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
DB_NORM_FACTOR = 100.0
DEFAULT_QUALITY = int(1)  # default video quality without agent
UPDATE_PER_EPOCH = 50  # update the parameters 8 times per epoch
RAND_RANGE = 1000
ENTROPY_EPS = 1e-6
A_DIM = 6
QUALITY_PENALTY = 0.8469011  # dB
REBUF_PENALTY = 28.79591348
SMOOTH_PENALTY_P = -0.29797156
SMOOTH_PENALTY_N = 1.06099887
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # kbps

# torch.cuda.set_device(0)
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor


def train_iml_vmaf(
    train_epoch, net_env, valid_env, args, video_files, add_str, summary_dir
):
    # Set-up output directories
    dt = datetime.datetime.now().strftime("%y%m%d_%H%M")
    net_desc = "{}_{}".format(dt, "_".join(args.name.split()))

    summary_dir_ = summary_dir + "/"
    save_folder = os.path.join(summary_dir_, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    writer = SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, "commandline_args.json")
        with open(params_path, "w") as f:
            json.dump(vars(args), f)

    log_file_name = save_folder + "/log"
    # command = 'rm ' + summary_dir + '/' + add_str + '/*'
    # os.system(command)
    # # logging.basicConfig(filename=log_file_name + '_central',
    # #                     filemode='w',
    # #                     level=logging.INFO)
    # writer = SummaryWriter(summary_dir + '/' + add_str + '/')

    video_size = {}  # in bytes
    vmaf_size = {}
    for bitrate in range(A_DIM):
        video_size[bitrate] = []
        vmaf_size[bitrate] = []
        with open(video_files[0] + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))
        with open(video_files[1] + str(bitrate)) as f:
            for line in f:
                vmaf_size[bitrate].append(float(line))

    latent_dim, mpc_horizon, gamma = args.latent_dim, args.mpc_h, args.gae_gamma
    kld_beta, kld_lambda, recon_gamma = args.kld_beta, args.kld_lambda, args.vae_gamma
    coeff_alpha, coeff_beta, coeff_gamma = 1, args.lc_beta, args.lc_gamma

    with open(log_file_name + "_record", "w") as log_file, open(
        log_file_name + "_test_im", "w"
    ) as test_log_file:
        torch.manual_seed(RANDOM_SEED)
        br_dim = A_DIM

        # Initialize the beta vae module
        vae_in_channels = 2  # 1 + 2 * br_dim
        c_len = 8
        vae_net = BetaVAE(
            in_channels=vae_in_channels,
            hist_dim=c_len,
            latent_dim=latent_dim,
            beta=kld_beta,
            delta=kld_lambda,
            gamma=recon_gamma,
        ).type(dtype)
        optimiser_vae = torch.optim.Adam(vae_net.parameters(), lr=LEARNING_RATE_VAE)

        # Initialize the rl agent
        # imitation learning with a latent-conditioned policy
        s_info, s_len = 17, 2
        total_chunk_num = 48
        model_actor = Actor(br_dim, latent_dim, s_info, s_len).type(dtype)
        optimizer_actor = optim.Adam(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)
        # optimizer_actor = optim.RMSprop(
        #                             model_actor.parameters(),
        #                             lr=LEARNING_RATE_ACTOR,
        #                             momentum=0.9)

        # load pre-train models
        if args.init:
            model_actor.load_state_dict(torch.load(args.actor_pt))
            vae_net.load_state_dict(torch.load(args.vae_pt))

        # --------------------- Interaction with environments -----------------------

        # define the observations for vae
        # observations for vae input
        ob = np.zeros((vae_in_channels, c_len))
        state = np.zeros((s_info, s_len))  # define the state for rl agent
        # state = torch.from_numpy(state)

        bit_rate_ = bit_rate_opt = last_bit_rate = DEFAULT_QUALITY
        time_stamp, end_flag = 0.0, True
        last_quality = vmaf_size[DEFAULT_QUALITY][0]

        # define the replay memory
        steps_in_episode, minibatch_size = args.ro_len, args.batch_size
        minibatch_size = minibatch_size
        epoch = 0
        memory = ReplayMemory(320 * steps_in_episode)

        max_QoE = {}
        max_QoE[0] = -99999
        # pdb.set_trace()

        for _ in range(train_epoch):
            # exploration in environments with expert strategy
            # memory.clear()
            model_actor.eval()
            vae_net.eval()
            states = []
            tar_actions = []
            obs = []
            rewards = []
            action_mask = np.ones(br_dim)
            for _ in range(steps_in_episode):
                # record the current state, observation and action
                if not end_flag:
                    states.append(state_)
                    obs.append(ob_)
                    tar_actions.append(
                        torch.from_numpy(np.array([bit_rate_opt])).type(dlongtype)
                    )

                # behavior policy, both use expert's trajectories and randomly trajectories

                bit_rate = bit_rate_

                # execute a step forward
                # delay, sleep_time, buffer_size, rebuf, \
                #     video_chunk_size, next_video_chunk_sizes, next_video_chunk_psnrs, \
                #         end_of_video, video_chunk_remain, _, curr_chunk_psnrs = \
                #             expert.step(bit_rate)
                net_env.get_video_chunk(bit_rate)
                (
                    delay,
                    sleep_time,
                    buffer_size,
                    rebuf,
                    video_chunk_size,
                    end_of_video,
                    video_chunk_remain,
                    video_chunk_vmaf,
                ) = (
                    net_env.delay0,
                    net_env.sleep_time0,
                    net_env.return_buffer_size0,
                    net_env.rebuf0,
                    net_env.video_chunk_size0,
                    net_env.end_of_video0,
                    net_env.video_chunk_remain0,
                    net_env.video_chunk_vmaf0,
                )

                next_video_chunk_sizes = []
                for i in range(A_DIM):
                    next_video_chunk_sizes.append(
                        video_size[i][net_env.video_chunk_counter]
                    )

                next_video_chunk_psnrs = []
                for i in range(A_DIM):
                    next_video_chunk_psnrs.append(
                        vmaf_size[i][net_env.video_chunk_counter]
                    )

                # ----compute and record the reward of current chunk ------
                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                curr_quality = video_chunk_vmaf
                sm_dif_p = max(curr_quality - last_quality, 0)
                sm_dif_n = max(last_quality - curr_quality, 0)
                reward = (
                    QUALITY_PENALTY * curr_quality
                    - REBUF_PENALTY * rebuf
                    - SMOOTH_PENALTY_P * sm_dif_p
                    - SMOOTH_PENALTY_N * sm_dif_n
                    - 2.661618558192494
                )

                # rewards.append(float(reward/REWARD_MAX))
                # reward_max = rebuffer_penalty
                # r_ = float(max(reward, -3*reward_max) / reward_max)
                rewards.append(reward)
                last_bit_rate = bit_rate
                last_quality = curr_quality

                # -------------- logging -----------------
                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(
                    str(time_stamp)
                    + "\t"
                    + str(VIDEO_BIT_RATE[bit_rate])
                    + "\t"
                    + str(VIDEO_BIT_RATE[bit_rate_opt])
                    + "\t"
                    + str(buffer_size)
                    + "\t"
                    + str(rebuf)
                    + "\t"
                    + str(video_chunk_size)
                    + "\t"
                    + str(delay)
                    + "\t"
                    + str(reward)
                    + "\n"
                )
                log_file.flush()

                ## dequeue history record
                state = np.roll(state, -1, axis=1)
                ob = np.roll(ob, -1, axis=1)

                # this should be S_INFO number of terms
                # state[0, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                state[0, -1] = (
                    float(video_chunk_size) / float(delay) / M_IN_K
                )  # kilo byte / ms
                state[1, -1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
                # last quality
                state[2, -1] = last_quality / DB_NORM_FACTOR
                state[3, -1] = np.minimum(video_chunk_remain, total_chunk_num) / float(
                    total_chunk_num
                )
                state[4 : 4 + br_dim, -1] = (
                    np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
                )  # mega byte
                state[10, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
                state[11 : 11 + br_dim, -1] = (
                    np.array(next_video_chunk_psnrs) / DB_NORM_FACTOR
                )

                ob[0, -1] = (
                    float(video_chunk_size) / float(delay) / M_IN_K
                )  # kilo byte / ms
                ob[1, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # seconds
                # ob[1 : 1 + br_dim, -1] = np.array(curr_chunk_sizes) / M_IN_K / M_IN_K # mega byte
                # ob[1 + br_dim : 1 + 2 * br_dim, -1] = np.array(curr_chunk_psnrs) / DB_NORM_FACTOR

                # bit_rate_opt = expert.optimal_action(args)
                net_env.get_optimal(float(last_quality))
                bit_rate_opt = int(net_env.optimal)

                ## find the feasible action space for the current chunk
                # curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
                # if ( len(past_bandwidth_ests) > 0 ):
                #     curr_error  = abs(past_bandwidth_ests[-1]-state[0,-1])/float(state[0,-1])
                # past_errors.append(curr_error)

                # cur_index = int(total_chunk_num - video_chunk_remain)
                # if cur_index == total_chunk_num:
                #     action_mask = np.ones(br_dim)
                #     future_bandwidth = 0
                # else:
                #     action_mask, future_bandwidth = \
                #         get_opt_space_vmaf_v1(
                #                         ob, state, last_quality, cur_index, bitrate_versions,
                #                         quality_penalty, rebuffer_penalty, smooth_penalty_p,
                #                         smooth_penalty_n, video_chunk_sizes, video_chunk_psnrs, past_errors)
                # past_bandwidth_ests.append(future_bandwidth)

                # compute the nn-based choice for next video chunk
                ob_ = np.array([ob]).transpose(0, 2, 1)  # [N, C_LEN, in_channels]
                ob_ = torch.from_numpy(ob_).type(dtype)

                state_ = np.array([state])
                state_ = torch.from_numpy(state_).type(dtype)

                action_mask_ = torch.from_numpy(action_mask).type(dtype)
                with torch.no_grad():
                    latent = vae_net.get_latent(ob_).detach()
                    prob = model_actor(state_, latent)
                    prob += 1e-5  ## with replacement=False, not enough non-negative category to sample if there is an output being zero
                    # prob = prob * action_mask_
                    action_ = prob.multinomial(num_samples=1)
                # selected = action_.squeeze().cpu().numpy()
                # del latent
                # del prob
                # while selected[0] >= np.sum(action_mask):
                #     selected = selected[1:]
                # bit_rate_ = selected[0]
                bit_rate_ = int(action_.squeeze().cpu().numpy())

                # retrieve the starting status
                end_flag = end_of_video
                if end_of_video:
                    # define the observations for vae
                    # observations for vae input
                    ob = np.zeros((vae_in_channels, c_len))

                    # define the state for rl agent
                    state = np.zeros((s_info, s_len))

                    bit_rate_ = bit_rate_opt = last_bit_rate = DEFAULT_QUALITY
                    time_stamp, end_flag = 0.0, True
                    last_quality = vmaf_size[DEFAULT_QUALITY][0]

                    # del past_bandwidth_ests[:]
                    # del past_errors[:]
                    log_file.write("\n")
                    log_file.flush()
                    break

            ##===store the transitions and learn the model===
            # compute returns and GAE(lambda) advantages:
            if len(states) != len(rewards):
                if len(states) + 1 == len(rewards):
                    rewards = rewards[1:]
                else:
                    print("error in length of states!")
                    break
            # R = Variable(R)

            # obs_target.append(np.zeros((1 + 2 * br_dim, c_len)))
            memory.push([states, tar_actions, obs])

            ##==Network parameters update==
            model_actor.train()
            vae_net.train()
            if memory.return_size() >= 1.2 * minibatch_size:
                # update the parameters
                vae_kld_loss_ = []
                policy_ce_loss_ = []
                policy_ent_loss_ = []
                policy_mi_loss_ = []
                for _ in range(steps_in_episode):
                    # sample minibatch from the replay memory
                    batch_states, batch_tar_actions, batch_obs = memory.sample_cuda(
                        minibatch_size
                    )
                    # states_size = np.shape(batch_states)
                    # action_size = np.shape(batch_tar_actions)
                    # obs_size = np.shape(batch_obs)
                    # assert states_size[1]==s_info and states_size[2]==s_len
                    # assert states_size[0] == action_size[0] and action_size[1] == 1
                    # assert obs_size[2] == c_len

                    ## learn the VAE network
                    # ------------------ VAE case -----------------------
                    batch_latents = vae_net.get_latent(batch_obs)

                    # latent samples for p(z_i|s)
                    # batch_s_ = batch_obs[:, -s_len:, :]

                    x_train = batch_obs  # (N, C_LEN, in_channels)

                    # fit the model
                    z_mu, z_log_var = vae_net.forward(x_train)
                    kld_loss = vae_net.loss_function(z_mu, z_log_var)
                    vae_kld_loss_.append(kld_loss.detach().cpu().numpy())

                    # record loss infors

                    ## learn the RL policy network
                    sample_num = args.sp_n
                    latent_samples = []
                    for _ in range(sample_num):
                        latent_samples.append(
                            torch.randn(minibatch_size, latent_dim).type(dtype)
                        )  # .detach()

                    ## compute actor loss (cross entropy loss, entropy loss, and mutual information loss)
                    batch_actions = batch_tar_actions.unsqueeze(1)
                    # pdb.set_trace()
                    probs_ = model_actor.forward(batch_states, batch_latents).clamp(
                        1e-4, 1.0 - 1e-4
                    )
                    prob_value_ = torch.gather(probs_, dim=1, index=batch_actions)
                    cross_entropy = -torch.mean(torch.log(prob_value_ + 1e-6))
                    entropy = -torch.mean(probs_ * torch.log(probs_ + 1e-6))

                    # mutual information loss
                    probs_samples = torch.zeros(minibatch_size, br_dim, 1).type(dtype)
                    for idx in range(sample_num):
                        probs_ = model_actor(batch_states, latent_samples[idx]).clamp(
                            1e-4, 1.0 - 1e-4
                        )
                        probs_ = probs_.unsqueeze(2)
                        probs_samples = torch.cat((probs_samples, probs_), 2)
                    probs_samples = probs_samples[:, :, 1:]
                    probs_sa = torch.mean(
                        probs_samples, dim=2
                    )  # p(a|s) = 1/L * \sum p(a|s, z_i) p(z_i|s)
                    probs_sa = Variable(probs_sa)
                    ent_noLatent = -torch.mean(probs_sa * torch.log(probs_sa + 1e-6))
                    mutual_info = ent_noLatent - entropy
                    loss_actor = -(
                        coeff_alpha * -1 * cross_entropy
                        + coeff_beta * entropy
                        + coeff_gamma * mutual_info
                    )
                    # loss_actor = - (cross_entropy + entropy + mutual_info)

                    # loss_actor = - (coeff_alpha * -1 * cross_entropy + coeff_beta * entropy)

                    optimizer_actor.zero_grad()
                    optimiser_vae.zero_grad()

                    ## compute the gradients
                    loss_total = loss_actor + kld_loss
                    loss_total.backward(retain_graph=False)

                    ## clip the gradients to aviod outputting inf or nan from the softmax function
                    # clip_grad_norm_(model_actor.parameters(), \
                    #                     max_norm = MAX_GRAD_NORM, norm_type = 2)
                    # clip_grad_norm_(vae_net.parameters(), \
                    #                     max_norm = MAX_GRAD_NORM, norm_type = 2)
                    ## update the parameters
                    optimizer_actor.step()
                    optimiser_vae.step()

                    # record the loss information
                    policy_ce_loss_.append(cross_entropy.detach().cpu().numpy())
                    policy_mi_loss_.append(mutual_info.detach().cpu().numpy())
                    policy_ent_loss_.append(entropy.detach().cpu().numpy())

                # ---- logging the loss information ----
                writer.add_scalar("Avg_loss_Ent", np.mean(policy_ent_loss_), epoch)
                writer.add_scalar("Avg_loss_CE", np.mean(policy_ce_loss_), epoch)
                writer.add_scalar("Avg_loss_MI", np.mean(policy_mi_loss_), epoch)
                writer.flush()

                # update epoch counter
                epoch += 1

            ## test and save the model
            if epoch % args.valid_i == 0 and epoch > 0:
                logging.info("Model saved in file")
                model_vae = vae_net
                mean_value = valid(
                    args,
                    valid_env,
                    model_actor,
                    model_vae,
                    epoch,
                    test_log_file,
                    save_folder,
                    add_str,
                )
                # entropy_weight = 0.95 * entropy_weight
                # ent_coeff = 0.95 * ent_coeff

                # save models
                save_models_v1(
                    logging,
                    save_folder,
                    add_str,
                    model_actor,
                    vae_net,
                    epoch,
                    max_QoE,
                    mean_value,
                )

                # turn on the training model
                model_actor.train()
                vae_net.train()

    writer.close()
    model_vae = vae_net
    return model_actor.state_dict(), model_vae.state_dict()


# def main():
#     train_bmpc(1000)

# if __name__ == '__main__':
#     main()
