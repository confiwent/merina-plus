import os
import torch
import numpy as np
import pdb

M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
TEST_TRACES_DIR = "./envs/traces"
TEST_LOG_DIR = "./Results/test"


def get_opt_space_vmaf(
    state,
    last_quality,
    index,
    br_versions,
    quality_p,
    rebuf_p,
    smooth_p,
    smooth_n,
    video_chunk_sizes,
    video_chunk_psnrs,
    past_errors,
):
    br_dim = len(br_versions)
    output = np.zeros(br_dim)
    output[0] = 1.0

    ## bandwidth prediction
    past_bandwidths = state[2, -5:]
    while past_bandwidths[0] == 0.0:
        past_bandwidths = past_bandwidths[1:]

    bandwidth_sum = 0
    for past_val in past_bandwidths:
        bandwidth_sum += 1 / float(past_val)
    harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
    # future_bandwidth = harmonic_bandwidth

    # future bandwidth prediction
    # divide by 1 + max of last 5 (or up to 5) errors
    max_error = 0
    error_pos = -5
    if len(past_errors) < 5:
        error_pos = -len(past_errors)
    max_error = float(max(past_errors[error_pos:]))
    future_bandwidth = harmonic_bandwidth / (1 + max_error)  # robustMPC here
    # future_bandwidth = harmonic_bandwidth

    ## last quality
    last_q = last_quality

    ## simulator infos
    # _, _, _, _,  = env.get_env_info()

    start_buffer = state[1, -1] * BUFFER_NORM_FACTOR

    ## Search feasible space
    for q in range(br_dim):
        output[q] = 1.0

        q_h = q + 1
        if q_h == br_dim:
            break

        ## quality values
        chunk_q_l = video_chunk_psnrs[q][index]
        chunk_q_h = video_chunk_psnrs[q_h][index]

        ## rebuffer time penalities
        download_time_l = (
            (video_chunk_sizes[q][index]) / 1000000.0 / future_bandwidth
        )  # this is MB/MB/s --> seconds
        # download_time_h = (video_chunk_sizes[q_h][index])/\
        #     1000000./future_bandwidth # this is MB/MB/s --> seconds

        rebuffer_time_l = max(download_time_l - start_buffer, 0)
        # rebuffer_time_h = max(download_time_h - start_buffer, 0)

        if rebuffer_time_l > 0:
            break

    return output, future_bandwidth


def get_test_traces(args):
    # configuration of test traces
    log_save_dir = os.path.join(*[TEST_LOG_DIR, args.res_folder])
    log_save_dir += "/"
    test_traces = os.path.join(*[TEST_TRACES_DIR, args.tr_folder])
    test_traces += "/"

    return log_save_dir, test_traces
