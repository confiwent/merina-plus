import argparse, time, sys, logging, os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from rl_test_vmaf import test

sys.path.append("./envs/")
import fixed_env_vmaf as env_test
import load_trace

S_INFO = 7  #
S_LEN = 8  # maximum length of states
C_LEN = 0  # content length
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # kbps
TOTAL_CHUNK_NUM = 49
QUALITY_PENALTY = 0.8469011  # dB
REBUF_PENALTY = 28.79591348
SMOOTH_PENALTY_P = -0.29797156
SMOOTH_PENALTY_N = 1.06099887

TEST_TRACES_DIR = "./envs/traces"
TEST_LOG_DIR = "./Results/test"

# use FCC and HSDPA datasets to jointly train the models
TRAIN_TRACES = "./envs/traces/pre_webget_1608/cooked_traces/"
VALID_TRACES = "./envs/traces/pre_webget_1608/test_traces/"

ADP_TRAIN_TRACES = "./envs/traces/puffer_211017/cooked_traces/"
ADP_VALID_TRACES = "./envs/traces/puffer_211017/test_traces/"

SUMMARY_DIR = "./Results/sim"
MODEL_DIR = "./saved_models"

# test models
TEST_MODEL_ACT_A2C = "./models/pensieve/actor_a2c_67000.model"
TEST_MODEL_ACT_ML = "./models/maml/maml_actor_1.pt"

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

parser = argparse.ArgumentParser(description="Neural_ABR")
parser.add_argument("--name", default="a2c", help="the name of result folder")
parser.add_argument("--a2c", action="store_true", help="Use pensieve")
parser.add_argument("--maml", action="store_true", help="Use A2BR")
parser.add_argument("--non-acp", action="store_true", help="Not use action pruning")
parser.add_argument("--res-folder", default="test", help="the name of result folder")
parser.add_argument("--tr-folder", default="test", help="the name of traces folder")


def get_test_traces(args):
    # print("Please choose the throughput data traces!!!")
    log_save_dir = os.path.join(*[TEST_LOG_DIR, args.res_folder])
    log_save_dir += "/"
    test_traces = os.path.join(*[TEST_TRACES_DIR, args.tr_folder])
    test_traces += "/"
    # log_save_dir = LOG_FILE
    # test_traces = TEST_TRACES

    log_path = log_save_dir + "log_test_" + args.name

    return log_save_dir, test_traces, log_path


def run_test(args, video_vmaf_file, video_size_file):
    log_save_dir, test_traces, log_path = get_test_traces(args)

    if not os.path.exists(log_save_dir):
        os.mkdir(log_save_dir)
    test_model_ = TEST_MODEL_ACT_ML if args.maml else TEST_MODEL_ACT_A2C

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)
    test_env = env_test.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        video_size_file=video_size_file,
        video_psnr_file=video_vmaf_file,
    )

    test_env.set_env_info(
        S_INFO,
        S_LEN,
        C_LEN,
        TOTAL_CHUNK_NUM,
        VIDEO_BIT_RATE,
        QUALITY_PENALTY,
        REBUF_PENALTY,
        SMOOTH_PENALTY_P,
        SMOOTH_PENALTY_N,
    )

    test(args, test_model_, test_env, log_path, log_save_dir)


def main():
    ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
    # parser = argparse.ArgumentParser()
    # _, rest_args = parser.parse_known_args()
    # args = args_maml.get_args(rest_args)
    args = parser.parse_args()
    video = "Mao"
    video_size_file = "./envs/video_size/" + video + "/video_size_"
    video_vmaf_file = "./envs/video_vmaf/chunk_vmaf"
    args.ckpt = f"{ts}"

    run_test(args, video_vmaf_file, video_size_file)


if __name__ == "__main__":
    main()
