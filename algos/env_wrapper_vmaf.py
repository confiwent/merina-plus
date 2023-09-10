import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from .helper import get_opt_space_vmaf_v2

DEFAULT_QUALITY = 1
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
DB_NORM_FACTOR = 100.0

torch.cuda.set_device(1)
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class VirtualPlayer:
    def __init__(self, args, env, log_file, vae_c=2):
        self.env = env
        self.args = args

        ## get the information of virtual players (personality)
        (
            s_info,
            s_len,
            c_len,
            total_chunk_num,
            bitrate_versions,
            quality_penalty,
            rebuffer_penalty,
            smooth_penalty_p,
            smooth_penalty_n,
        ) = env.get_env_info()

        # QoE metrics
        self.quality_p = quality_penalty
        self.rebuff_p = rebuffer_penalty
        self.smooth_p = smooth_penalty_p
        self.smooth_n = smooth_penalty_n

        # Video information
        self.total_chunk_num = total_chunk_num
        self.bitrate_versions = bitrate_versions
        self.br_dim = len(bitrate_versions)

        # states and observation configures
        self.c_len = c_len
        self.s_info = s_info
        self.s_len = s_len

        # define the observations for vae
        self.vae_in_channels = vae_c
        self.ob = np.zeros((self.vae_in_channels, self.c_len))  # observations
        # define the state for rl agent
        self.state = np.zeros((self.s_info, self.s_len))

        # information of emulating the video playing
        self.last_bit_rate = DEFAULT_QUALITY
        self.last_quality = self.env.chunk_psnr[DEFAULT_QUALITY][0]
        self.time_stamp = 0.0
        self.end_flag = True
        self.video_chunk_remain = self.total_chunk_num

        # log files, recoding the video playing
        self.log_file = log_file

        # information of action mask
        self.past_errors = []
        self.past_bandwidth_ests = []
        self.video_chunk_sizes = env.get_video_size()
        self.video_chunk_psnrs = env.get_video_psnr()

        self.action_mask = np.ones(self.br_dim)

        ## define the reward normalization class
        # self.reward_filter = Identity()
        # self.reward_filter = ZFilter(self.reward_filter, shape=(), center=False, clip=10.)
        # self.reward_filter = RewardFilter(self.reward_filter, shape=(), gamma=0.95, clip=10.)

    def step(self, bit_rate):
        # execute a step forward
        (
            delay,
            sleep_time,
            buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            next_video_chunk_psnrs,
            end_of_video,
            video_chunk_remain,
            _,
            curr_chunk_psnrs,
        ) = self.env.get_video_chunk(bit_rate)

        # compute and record the reward of current chunk
        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        self.video_chunk_remain = video_chunk_remain

        # compute reward of Quality of experience
        curr_quality = curr_chunk_psnrs[bit_rate]
        sm_dif_p = max(curr_quality - self.last_quality, 0)
        sm_dif_n = max(self.last_quality - curr_quality, 0)
        reward = (
            self.quality_p * curr_quality
            - self.rebuff_p * rebuf
            - self.smooth_p * sm_dif_p
            - self.smooth_n * sm_dif_n
            - 2.661618558192494
        )

        rew_ = float(max(reward, -6 * self.rebuff_p) / 100.0)
        # reward_norm = self.reward_filter(rew_)
        reward_norm = rew_

        self.last_bit_rate = bit_rate
        self.last_quality = curr_quality

        # -------------- logging -----------------
        # log time_stamp, bit_rate, buffer_size, reward
        self.log_file.write(
            str(self.time_stamp)
            + "\t"
            + str(self.bitrate_versions[bit_rate])
            + "\t"
            +
            # str(np.sum(self.action_mask)) + '\t' +
            str(buffer_size)
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
        self.log_file.flush()

        ## dequeue history record
        self.state = np.roll(self.state, -1, axis=1)
        self.ob = np.roll(self.ob, -1, axis=1)

        # this should be S_INFO number of terms
        # state[0, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        self.state[0, -1] = (
            float(video_chunk_size) / float(delay) / M_IN_K
        )  # kilo bytes / ms
        self.state[1, -1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
        self.state[2, -1] = self.last_quality / DB_NORM_FACTOR  # last quality
        self.state[3, -1] = np.minimum(
            self.video_chunk_remain, self.total_chunk_num
        ) / float(self.total_chunk_num)
        self.state[4 : 4 + self.br_dim, -1] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        )  # mega byte
        self.state[10, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        self.state[11 : 11 + self.br_dim, -1] = (
            np.array(next_video_chunk_psnrs) / DB_NORM_FACTOR
        )

        self.ob[0, -1] = (
            float(video_chunk_size) / float(delay) / M_IN_K
        )  # kilo bytes / ms
        self.ob[1, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # seconds

        action_mask_ = self.get_action_mask()

        ob_ = np.array([self.ob]).transpose(0, 2, 1)  # [N, C_LEN, in_channels]
        ob_ = torch.from_numpy(ob_).type(dtype)

        state_ = np.array([self.state])
        state_ = torch.from_numpy(state_).type(dtype)

        self.end_flag = end_of_video
        if self.end_flag:
            self.reset_play()

        return ob_, state_, reward_norm, end_of_video, action_mask_
        # return ob_, state_, reward_norm, end_of_video

    def get_action_mask(self):
        ##find the feasible action space for the current chunk
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if len(self.past_bandwidth_ests) > 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - self.state[0, -1]) / float(
                self.state[0, -1]
            )
        self.past_errors.append(curr_error)

        cur_index = int(self.total_chunk_num - self.video_chunk_remain)
        if cur_index == self.total_chunk_num:
            self.action_mask = np.ones(self.br_dim)
            future_bandwidth = 0
        else:
            self.action_mask, future_bandwidth = get_opt_space_vmaf_v2(
                self.ob,
                self.state,
                self.last_quality,
                cur_index,
                self.bitrate_versions,
                self.quality_p,
                self.rebuff_p,
                self.smooth_p,
                self.smooth_n,
                self.video_chunk_sizes,
                self.video_chunk_psnrs,
                self.past_errors,
            )
        action_mask_ = torch.from_numpy(self.action_mask).type(dtype)

        self.past_bandwidth_ests.append(future_bandwidth)

        return action_mask_

    def reset_play(self):
        self.ob = np.zeros((self.vae_in_channels, self.c_len))  # observations
        self.state = np.zeros((self.s_info, self.s_len))

        self.last_bit_rate = DEFAULT_QUALITY
        self.last_quality = self.env.chunk_psnr[DEFAULT_QUALITY][0]
        self.video_chunk_remain = self.total_chunk_num
        self.time_stamp = 0.0

        self.past_bandwidth_ests = []
        self.past_errors = []
        self.action_mask = np.ones(self.br_dim)

        self.log_file.write("\n")
        self.log_file.flush()

    # def reset_reward(self):
    #     self.reward_filter.reset()
