"""
In this version, the RobustMPC is adopted to control the rate adaptation 
with the harmonic mean bandwidth prediction method. 
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import itertools
import time
import argparse
from helper_baseline import get_test_traces

sys.path.append("./envs/")
import fixed_env_vmaf as env
import load_trace

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 3
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 49.0
TOTAL_VIDEO_CHUNKS = 49
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # dB
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000

parser = argparse.ArgumentParser(description="RobustMPC")
parser.add_argument("--res-folder", default="test", help="the name of result folder")
parser.add_argument("--tr-folder", default="puffer", help="the name of traces folder")

CHUNK_COMBO_OPTIONS = []
# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []


def main():
    start = time.time()

    args = parser.parse_args()  # ["--tf"]
    video = "Mao"
    # video = 'Avengers'
    video_size_file = "./envs/video_size/" + video + "/video_size_"
    video_vmaf_file = "./envs/video_vmaf/chunk_vmaf"

    log_save_dir, test_traces = get_test_traces(args)

    if not os.path.exists(log_save_dir):
        os.mkdir(log_save_dir)

    log_file_init = log_save_dir + "log_test_mpc"

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)
    test_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        video_size_file=video_size_file,
        video_psnr_file=video_vmaf_file,
    )

    test_env.set_env_info(
        0,
        0,
        0,
        int(CHUNK_TIL_VIDEO_END_CAP),
        VIDEO_BIT_RATE,
        1,
        REBUF_PENALTY,
        SMOOTH_PENALTY,
        0,
    )  # the QoE weights don't matter here!

    log_path = log_file_init + "_" + all_file_names[test_env.trace_idx]
    log_file = open(log_path, "w")

    _, _, _, total_chunk_num, bitrate_versions, _, _, _, _ = test_env.get_env_info()

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # chunk_size_info = video_size()
    # chunk_size_info.store_size(video_size_file)

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    last_chunk_vmaf = test_env.chunk_psnr[DEFAULT_QUALITY][0]
    harmonic_bandwidth = 0
    future_bandwidth = 0

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    # entropy_record = []

    # make chunk combination options
    for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=MPC_FUTURE_CHUNK_COUNT):
        CHUNK_COMBO_OPTIONS.append(combo)

    video_count = 0
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        (
            delay,
            sleep_time,
            buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            next_video_chunk_vmaf,
            end_of_video,
            video_chunk_remain,
            _,
            curr_chunk_vmafs,
        ) = test_env.get_video_chunk(bit_rate)

        video_chunk_vmaf = curr_chunk_vmafs[bit_rate]

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = (
            0.8469011 * video_chunk_vmaf
            - 28.79591348 * rebuf
            + 0.29797156 * np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.0))
            - 1.06099887 * np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.0))
            - 2.661618558192494
        )
        r_batch.append(reward)

        sm_dif_p = np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.0))
        sm_dif_n = np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.0))

        last_bit_rate = bit_rate
        last_chunk_vmaf = video_chunk_vmaf

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(
            str(time_stamp / M_IN_K)
            + "\t"
            + str(VIDEO_BIT_RATE[bit_rate])
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
            + "\t"
            + str(video_chunk_vmaf)
            + "\t"
            + str(sm_dif_p)
            + "\t"
            + str(sm_dif_n)
            + "\n"
        )
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
            np.max(VIDEO_BIT_RATE)
        )  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP
        )
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if len(past_bandwidth_ests) > 0:
            curr_error = abs(past_bandwidth_ests[-1] - state[3, -1]) / float(
                state[3, -1]
            )
        past_errors.append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3, -5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        # if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        # else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += 1 / float(past_val)
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if len(past_errors) < 5:
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth / (1 + max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)
        # past_bandwidth_ests.append(future_bandwidth)

        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if TOTAL_VIDEO_CHUNKS - last_index - 1 < MPC_FUTURE_CHUNK_COUNT:
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index - 1

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = buffer_size
        # start = time.time()
        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs_p = 0
            smoothness_diffs_n = 0
            last_bit_rate_ = last_bit_rate
            last_chunk_vmaf_ = last_chunk_vmaf
            for position in range(0, len(combo)):
                index = (
                    last_index + position + 1
                )  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                chunk_action = combo[position]
                chunk_quality_ = test_env.chunk_psnr[chunk_action][index]
                download_time = (
                    test_env.video_size[chunk_action][index] / 1000000.0
                ) / future_bandwidth  # this is MB/MB/s --> seconds
                if curr_buffer < download_time:
                    curr_rebuffer_time += download_time - curr_buffer
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += chunk_quality_
                # smoothness_diffs_p += max(chunk_quality_ - last_quality_, 0)
                # smoothness_diffs_n += max(last_quality_ - chunk_quality_, 0)
                # bitrate_sum += BITRATE_REWARD[chunk_quality]
                last_quality_ = last_chunk_vmaf_
                smoothness_diffs_p += np.abs(
                    np.maximum(chunk_quality_ - last_quality_, 0.0)
                )
                smoothness_diffs_n = np.abs(
                    np.minimum(chunk_quality_ - last_quality_, 0.0)
                )
                last_bit_rate_ = chunk_action
                last_chunk_vmaf_ = chunk_quality_
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

            reward = (
                0.8469011 * bitrate_sum
                - (28.79591348 * curr_rebuffer_time)
                + 0.29797156 * smoothness_diffs_p
                - 1.06099887 * smoothness_diffs_n
            )

            if reward >= max_reward:
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = (
                    0  # no combo had reward better than -1000000 (ERROR) so send 0
                )
                if best_combo != ():  # some combo was good
                    send_data = best_combo[0]

        bit_rate = send_data
        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

        if end_of_video:
            log_file.write("\n")
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            last_chunk_vmaf = test_env.chunk_psnr[DEFAULT_QUALITY][0]

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            del past_bandwidth_ests[:]
            del past_errors[:]

            time_stamp = 0

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                end = time.time()
                print(end - start)
                break

            log_path = log_file_init + "_" + all_file_names[test_env.trace_idx]
            log_file = open(log_path, "w")

            end = time.time()
            print(end - start)


if __name__ == "__main__":
    main()
