"""
Using VAMF as the quality metric

Imitation learning pre-training

For journal version. 2023

copyright@Kan, Nuowen, kannw_1230@sjtu.edu.cn

"""

import os
from tqdm import tqdm
import argparse
import pdb
import datetime
import torch

from config import args_merina_vmaf_j
from algos.test_vmaf import test
from algos.train_im_vmaf import train_iml_vmaf
from algos.rl_training_vmaf import ppo_training
import envs.env_vmaf as env
import envs.envcpp as env_im
import envs.fixed_env_vmaf as env_test
from envs import load_trace

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Parameters of envs
S_INFO = 17  #
S_LEN = 2  # maximum length of states
C_LEN = 8  # content length
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

ADP_TRAIN_TRACES = "./envs/traces/puffer_adp_0210/cooked_traces/"
ADP_VALID_TRACES = "./envs/traces/puffer_adp_0210/test_traces/"

SUMMARY_DIR = "./Results/sim"
MODEL_DIR = "./saved_models"

# test models
TEST_MODEL_ACT_IL = "./models/230312_2236/policy_merinaJ_500.model"
TEST_MODEL_VAE_IL = "./models/230312_2236/VAE_merinaJ_500.model"

TEST_MODEL_ACT_IL_NMI = (
    "./models/0502_imrl/policy_oil_500.model"  #'./save_models/oil/Policy_oil_550.model'
)
TEST_MODEL_VAE_IL_NMI = (
    "./models/0502_imrl/VAE_oil_500.model"  #'./save_models/oil/VAE_oil_5050.model'
)

TEST_MODEL_ACT_MRL_NAP = (
    "./models/0502_imrl/policy_0502imrl_2000.model"  # 12/300 is good
)
TEST_MODEL_VAE_MRL_NAP = "./models/0502_imrl/VAE_0502imrl_2000.model"  # for mm22

# TEST_MODEL_ACT_MRL = './models/0502_imrl/policy_imrl_400.model' # for mm22
# TEST_MODEL_VAE_MRL = './models/0502_imrl/VAE_imrl_400.model'
TEST_MODEL_ACT_MRL = "./models/ema_0508/policy_imrl_1250.model"
TEST_MODEL_VAE_MRL = "./models/ema_0508/VAE_imrl_1250.model"


def main():
    parser = argparse.ArgumentParser()
    _, rest_args = parser.parse_known_args()
    args = args_merina_vmaf_j.get_args(rest_args)

    video_size_file = "./envs/video_size/Mao/video_size_"  # video = 'origin'
    video_vmaf_file = "./envs/video_vmaf/chunk_vmaf"

    if args.test:
        run_test(args, video_vmaf_file, video_size_file)
    else:
        run_train(args, video_vmaf_file, video_size_file)


def get_test_traces(args):
    # configuration of test traces
    log_save_dir = os.path.join(*[TEST_LOG_DIR, args.res_folder])
    log_save_dir += "/"
    test_traces = os.path.join(*[TEST_TRACES_DIR, args.tr_folder])
    test_traces += "/"

    log_path = log_save_dir + "log_test_" + args.name

    return log_save_dir, test_traces, log_path


def get_models(args):
    if args.mm22:
        return [TEST_MODEL_ACT_MRL_NAP, TEST_MODEL_VAE_MRL_NAP]
    elif args.il:
        return [TEST_MODEL_ACT_IL, TEST_MODEL_VAE_IL]
    elif args.nmi:
        return [TEST_MODEL_ACT_IL_NMI, TEST_MODEL_VAE_IL_NMI]
    else:
        return [TEST_MODEL_ACT_MRL, TEST_MODEL_VAE_MRL]


def run_test(args, video_vmaf_file, video_size_file):
    log_save_dir, test_traces, log_path = get_test_traces(args)

    if not os.path.exists(log_save_dir):
        os.mkdir(log_save_dir)

    test_model_ = get_models(args)

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


def im_training(args, video_files, train_env, valid_env, log_dir_path):
    add_str = args.name
    train_epochs = args.epochs
    model_actor_para, model_vae_para = train_iml_vmaf(
        train_epochs, train_env, valid_env, args, video_files, add_str, log_dir_path
    )

    # ##===== save models in the First stage
    model_save_dir = MODEL_DIR + "/" + add_str
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    # command = 'rm ' + SUMMARY_DIR + add_str + '/*'5
    # os.system(command)
    model_actor_save_path = model_save_dir + "/%s_%s_%d.model" % (
        str("Policy"),
        add_str,
        int(train_epochs),
    )
    model_vae_save_path = model_save_dir + "/%s_%s_%d.model" % (
        str("VAE"),
        add_str,
        int(train_epochs),
    )
    if os.path.exists(model_actor_save_path):
        os.system("rm " + model_actor_save_path)
    if os.path.exists(model_vae_save_path):
        os.system("rm " + model_vae_save_path)
    torch.save(model_actor_para, model_actor_save_path)
    torch.save(model_vae_para, model_vae_save_path)

    ## COPY THE LOG FILE
    # os.system('cp ' + log_dir_path + '/' + add_str + '/log_test ' + model_save_dir + '/')

    return model_actor_save_path, model_vae_save_path


def run_train(args, video_vmaf_file, video_size_file):
    add_str = args.name
    log_dir_path = SUMMARY_DIR
    video_files = [video_size_file, video_vmaf_file]

    ##=== environments configures============
    if args.adp:
        # Train_traces = ADP_TRAIN_TRACES
        # Valid_traces = ADP_VALID_TRACES
        trace_folder = args.tr_folder
        Train_traces = os.path.join(*[TEST_TRACES_DIR, trace_folder, "cooked_traces/"])
        Valid_traces = os.path.join(*[TEST_TRACES_DIR, trace_folder, "test_traces/"])
    else:
        Train_traces = TRAIN_TRACES
        Valid_traces = VALID_TRACES
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(Valid_traces)
    valid_env = env_test.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        video_size_file=video_size_file,
        video_psnr_file=video_vmaf_file,
    )
    valid_env.set_env_info(
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

    im_train_env = env_im.Environment(Train_traces)

    ppo_train_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        video_size_file=video_size_file,
        video_psnr_file=video_vmaf_file,
    )
    ppo_train_env.set_env_info(
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

    if args.init:
        args.actor_pt = "./models/230502_1535/policy_iml_1000.model"
        args.vae_pt = "./models/230502_1535/VAE_iml_1000.model"

    if args.adp and not args.fscra:
        if args.from_il:
            model_actor_save_path = "./saved_models/scratch/Policy_scratch_550.model"
            model_vae_save_path = "./saved_models/scratch/VAE_scratch_550.model"
            print("IL initial models have been loaded!")
        else:
            model_actor_save_path = "./models/adp_init/policy_imrl_1250.model"
            model_vae_save_path = "./models/adp_init/VAE_imrl_1250.model"
            model_critic_save_path = "./models/adp_init/critic_imrl_1250.model"
            print("PPO initial models have been loaded!")
    elif args.from_il:
        model_actor_save_path = "./saved_models/scratch/Policy_scratch_550.model"
        model_vae_save_path = "./saved_models/scratch/VAE_scratch_550.model"
        print("IL initial models have been loaded!")
    else:
        model_actor_save_path, model_vae_save_path = im_training(
            args, video_files, im_train_env, valid_env, log_dir_path
        )

    # RL part
    model_vae_para = torch.load(model_vae_save_path)
    model_actor_para = torch.load(model_actor_save_path)
    model_critic_para = None if args.vap else torch.load(model_critic_save_path)
    # model_critic_para = None

    ppo_training(
        model_actor_para,
        model_vae_para,
        model_critic_para,
        ppo_train_env,
        valid_env,
        args,
        add_str,
        log_dir_path,
    )


if __name__ == "__main__":
    main()
