import argparse
import sys
import os
import numpy as np
from helper_baseline import get_test_traces

sys.path.append("./envs/")
import fixed_env_vmaf as env
import load_trace


A_DIM = 6
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
M_IN_K = 1000.0
CHUNK_TIL_VIDEO_END_CAP = 49.0
REBUF_PENALTY = 4.3  # dB
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
MINIMUM_BUFFER_S = 8
BUFFER_TARGET_S = 20

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
parser = argparse.ArgumentParser(description="BOLA")
parser.add_argument("--res-folder", default="test", help="the name of result folder")
parser.add_argument("--tr-folder", default="puffer", help="the name of traces folder")


def main():
    args = parser.parse_args()

    video = "Mao"
    # video = 'Avengers'
    video_size_file = "./envs/video_size/" + video + "/video_size_"
    video_vmaf_file = "./envs/video_vmaf/chunk_vmaf"

    log_save_dir, test_traces = get_test_traces(args)

    if not os.path.exists(log_save_dir):
        os.mkdir(log_save_dir)

    log_file_init = log_save_dir + "log_test_bola"

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

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    last_chunk_vmaf = test_env.chunk_psnr[DEFAULT_QUALITY][0]
    bit_rate = DEFAULT_QUALITY

    r_batch = []
    # gp = 1 - 0 + (np.log(VIDEO_BIT_RATE[-1] / float(VIDEO_BIT_RATE[0])) - 0) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # log
    # vp = MINIMUM_BUFFER_S/(0+ gp -1)
    # gp = 1 - VIDEO_BIT_RATE[0]/1000.0 + (VIDEO_BIT_RATE[-1]/1000. - VIDEO_BIT_RATE[0]/1000.) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # lin
    # vp = MINIMUM_BUFFER_S/(VIDEO_BIT_RATE[0]/1000.0+ gp -1)

    video_count = 0
    vmaf_min = []
    vmaf_max = []

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
        vmaf_max.append(curr_chunk_vmafs[-1])
        vmaf_min.append(curr_chunk_vmafs[0])

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

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

        # if buffer_size < RESEVOIR:
        #     bit_rate = 0
        # elif buffer_size >= RESEVOIR + CUSHION:
        #     bit_rate = A_DIM - 1
        # else:
        #     bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        # q_min = np.mean(vmaf_min)
        # q_max = np.mean(vmaf_max)
        q_min = vmaf_min[-1] / 100
        q_max = vmaf_max[-1] / 100
        gp = (
            1 - q_min + (q_max - q_min) / (BUFFER_TARGET_S / MINIMUM_BUFFER_S - 1)
        )  # log
        vp = MINIMUM_BUFFER_S / (q_min + gp - 1)
        # gp = 1 - 0 + (np.log(VIDEO_BIT_RATE[-1] / float(VIDEO_BIT_RATE[0])) - 0) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # log
        # vp = MINIMUM_BUFFER_S/(0+ gp -1)

        score = -65535
        for q in range(len(VIDEO_BIT_RATE)):
            q_c = curr_chunk_vmafs[q] / 100
            s = (vp * (q_c + gp) - buffer_size) / next_video_chunk_sizes[q]
            # s = (vp * (VIDEO_BIT_RATE[q]/1000. + gp) - buffer_size) / next_video_chunk_sizes[q] # lin
            # s = (vp * (np.log(VIDEO_BIT_RATE[q] / float(VIDEO_BIT_RATE[0])) + gp) - buffer_size) / next_video_chunk_sizes[q]
            if s >= score:
                score = s
                bit_rate = q

        bit_rate = int(bit_rate)

        if end_of_video:
            log_file.write("\n")
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            last_chunk_vmaf = test_env.chunk_psnr[DEFAULT_QUALITY][0]
            bit_rate = DEFAULT_QUALITY  # use the default action here

            time_stamp = 0

            print("video count", video_count)
            video_count += 1
            vmaf_min = []
            vmaf_max = []

            if video_count > len(all_file_names):
                break

            log_path = log_file_init + "_" + all_file_names[test_env.trace_idx]
            log_file = open(log_path, "w")


if __name__ == "__main__":
    main()
