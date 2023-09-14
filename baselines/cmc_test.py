import os
import sys, time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import libcomyco

sys.path.append("./envs/")
import fixed_env_vmaf as env
import load_trace


S_INFO = 7  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 49.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
# LOG_FILE = './Results/test/log_sim_cmc'
# TEST_TRACES = './cooked_test_traces/'
# # log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = sys.argv[1]
NN_MODEL = "./models/cmc/nn_model_ep_330.ckpt"  # for vmaf

TEST_TRACES_DIR = "./envs/traces"
TEST_LOG_DIR = "./Results/test"

parser = argparse.ArgumentParser(description="Comyco-JSAC20")
parser.add_argument("--res-folder", default="test", help="the name of result folder")
parser.add_argument("--tr-folder", default="puffer", help="the name of traces folder")


def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    args = parser.parse_args()
    video = "Mao"
    # video = 'Avengers'
    video_size_file = "./envs/video_size/" + video + "/video_size_"
    video_vmaf_file = "./envs/video_vmaf/chunk_vmaf"

    # --------- initialize the environment-------------
    # print("Please choose the throughput data traces!!!")
    log_save_dir = os.path.join(*[TEST_LOG_DIR, args.res_folder])
    log_save_dir += "/"
    test_traces = os.path.join(*[TEST_TRACES_DIR, args.tr_folder])
    test_traces += "/"

    if not os.path.exists(log_save_dir):
        os.mkdir(log_save_dir)

    log_file_init = log_save_dir + "log_test_cmc"

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)

    net_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        video_size_file=video_size_file,
        video_psnr_file=video_vmaf_file,
    )

    net_env.set_env_info(
        0,
        0,
        0,
        int(CHUNK_TIL_VIDEO_END_CAP),
        VIDEO_BIT_RATE,
        1,
        REBUF_PENALTY,
        SMOOTH_PENALTY,
        0,
    )

    log_path = log_file_init + "_" + all_file_names[net_env.trace_idx]
    log_file = open(log_path, "w")

    _, _, _, total_chunk_num, bitrate_versions, _, _, _, _ = net_env.get_env_info()

    with tf.Session() as sess:
        actor = libcomyco.libcomyco(sess, S_INFO, S_LEN, A_DIM, LR_RATE=1e-4)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = 0

        bit_rate = DEFAULT_QUALITY
        last_chunk_vmaf = None

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        start_time = time.time()

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
            ) = net_env.get_video_chunk(bit_rate)

            video_chunk_vmaf = curr_chunk_vmafs[bit_rate]

            if last_chunk_vmaf is None:
                last_chunk_vmaf = video_chunk_vmaf

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            reward = (
                0.8469011 * video_chunk_vmaf
                - 28.79591348 * rebuf
                + 0.29797156
                * np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.0))
                - 1.06099887
                * np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.0))
                - 2.661618558192494
            )
            r_batch.append(reward)

            sm_dif_p = np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.0))
            sm_dif_n = np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.0))

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
            state[0, -1] = video_chunk_vmaf / 100.0
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = (
                float(video_chunk_size) / float(delay) / M_IN_K
            )  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = (
                np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
            )  # mega byte
            state[5, :A_DIM] = np.array(next_video_chunk_vmaf) / 100.0  # mega byte
            state[6, -1] = np.minimum(
                video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP
            ) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob, _ = actor.predict(np.reshape(state, (-1, S_INFO, S_LEN)))
            bit_rate = np.argmax(action_prob[0])

            s_batch.append(state)

            entropy_record.append(actor.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write("\n")
                log_file.close()

                bit_rate = DEFAULT_QUALITY  # use the default action here
                last_chunk_vmaf = None

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = log_file_init + "_" + all_file_names[net_env.trace_idx]
                log_file = open(log_path, "w")

        end_time = time.time()
        print("Running time:")
        print(end_time - start_time)
        # printf()


if __name__ == "__main__":
    main()
