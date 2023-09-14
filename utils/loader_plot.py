import os, sys, json
import numpy as np
import pandas as pd

MILLISEC_IN_SEC = 1000.0
M_IN_K = 1000.0
M_IN_B = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
QUALITY_PENALTY = 0.8469011  # dB
REBUF_PENALTY = 28.79591348
SMOOTH_PENALTY_P = -0.29797156
SMOOTH_PENALTY_N = 1.06099887


def read_dict_from_json(path):
    with open(path, "r") as f:
        dict = f.read()
    return json.loads(dict)


def load_source_settings(args, data_dir, pic_dir, lbl_path):
    data_dict = {}
    lbl_dict = read_dict_from_json(lbl_path)
    if args.tf:
        lbl = lbl_dict["fcc"][0]
        data_dict[lbl] = [os.path.join(*[data_dir, lbl]), os.path.join(*[pic_dir, lbl])]

    if args.t3g:
        lbl = lbl_dict["3gp"][0]
        data_dict[lbl] = [os.path.join(*[data_dir, lbl]), os.path.join(*[pic_dir, lbl])]

    if args.tfh:
        lbl = lbl_dict["fh"][0]
        data_dict[lbl] = [os.path.join(*[data_dir, lbl]), os.path.join(*[pic_dir, lbl])]

    if args.to:
        lbl = lbl_dict["oboe"][0]
        data_dict[lbl] = [os.path.join(*[data_dir, lbl]), os.path.join(*[pic_dir, lbl])]

    if args.tp:
        lbl = lbl_dict["puffer1"][0]
        data_dict[lbl] = [os.path.join(*[data_dir, lbl]), os.path.join(*[pic_dir, lbl])]

    if args.tp2:
        lbl = lbl_dict["puffer2"][0]
        data_dict[lbl] = [os.path.join(*[data_dir, lbl]), os.path.join(*[pic_dir, lbl])]

    if args.tp3:
        lbl = lbl_dict["puffer3"][0]
        data_dict[lbl] = [os.path.join(*[data_dir, lbl]), os.path.join(*[pic_dir, lbl])]

    return data_dict


def load_labels(args, lbl_path):
    scm_list = []
    lbl_dic = read_dict_from_json(lbl_path)
    if args.bola:
        name, label = lbl_dic["bola"][0], lbl_dic["bola"][1]
        scm_list.append([name, label])

    if args.mpc:
        name, label = lbl_dic["mpc"][0], lbl_dic["mpc"][1]
        scm_list.append([name, label])

    if args.pensieve:
        name, label = lbl_dic["pensieve"][0], lbl_dic["pensieve"][1]
        scm_list.append([name, label])

    if args.comyco:
        name, label = lbl_dic["comyco"][0], lbl_dic["comyco"][1]
        scm_list.append([name, label])

    if args.fugo:
        name, label = lbl_dic["fugo"][0], lbl_dic["fugo"][1]
        scm_list.append([name, label])

    if args.bayes:
        name, label = lbl_dic["bayes"][0], lbl_dic["bayes"][1]
        scm_list.append([name, label])

    if args.maml:
        name, label = lbl_dic["maml"][0], lbl_dic["maml"][1]
        scm_list.append([name, label])

    if args.mmrl:
        name, label = lbl_dic["mmrl"][0], lbl_dic["mmrl"][1]
        scm_list.append([name, label])

    if args.iml:
        name, label = lbl_dic["iml"][0], lbl_dic["iml"][1]
        scm_list.append([name, label])

    if args.nacrl:
        name, label = lbl_dic["nacrl"][0], lbl_dic["nacrl"][1]
        scm_list.append([name, label])

    if args.nmirl:
        name, label = lbl_dic["nmirl"][0], lbl_dic["nmirl"][1]
        scm_list.append([name, label])

    if args.imrl:
        name, label = lbl_dic["imrl"][0], lbl_dic["imrl"][1]
        scm_list.append([name, label])

    if args.adp:
        name, label = lbl_dic["adp"][0], lbl_dic["adp"][1]
        scm_list.append([name, label])

    return scm_list


def load_data_from_log(scm_key, res_folder):
    time_all = {}  # checkpoint
    bit_rate_all = {}  # bitrate
    buff_all = {}  # buffer
    rebuf_all = {}  # rebuffering time
    bw_all = {}  # throughput value
    raw_reward_all = {}  # reward
    quality_all = {}  # vamf quality of chunk
    sm_p_all = {}  # smoothness penalty positive
    sm_n_all = {}  # smoothness penalty negative

    for scheme in scm_key:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        bit_rate_all[scheme] = {}
        buff_all[scheme] = {}
        rebuf_all[scheme] = {}
        bw_all[scheme] = {}
        quality_all[scheme] = {}
        sm_p_all[scheme] = {}
        sm_n_all[scheme] = {}

    log_files = os.listdir(res_folder)
    for log_file in log_files:
        time_ms = []
        bit_rate = []
        buff = []
        rebuf = []
        bw = []
        reward = []
        quality = []
        sm_p = []
        sm_n = []

        file_path = os.path.join(*[res_folder, log_file])
        with open(file_path, "rb") as f:
            for line in f:
                parse = line.split()
                if len(parse) <= 1:
                    break
                time_ms.append(float(parse[0]))
                bit_rate.append(int(parse[1]))
                buff.append(float(parse[2]))
                rebuf.append(float(parse[3]))
                bw.append(
                    float(parse[4])
                    / float(parse[5])
                    * BITS_IN_BYTE
                    * MILLISEC_IN_SEC
                    / M_IN_B
                )
                reward.append(float(parse[6]))
                quality.append(float(parse[7]))
                sm_p.append(float(parse[8]))
                sm_n.append(float(parse[9]))

        # start the record from t=0
        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]

        for scheme in scm_key:
            if scheme in log_file:
                time_all[scheme][log_file[len("log_" + str(scheme) + "_") :]] = time_ms
                bit_rate_all[scheme][
                    log_file[len("log_" + str(scheme) + "_") :]
                ] = bit_rate
                buff_all[scheme][log_file[len("log_" + str(scheme) + "_") :]] = buff
                rebuf_all[scheme][log_file[len("log_" + str(scheme) + "_") :]] = rebuf
                bw_all[scheme][log_file[len("log_" + str(scheme) + "_") :]] = bw
                raw_reward_all[scheme][
                    log_file[len("log_" + str(scheme) + "_") :]
                ] = reward
                quality_all[scheme][
                    log_file[len("log_" + str(scheme) + "_") :]
                ] = quality
                sm_p_all[scheme][log_file[len("log_" + str(scheme) + "_") :]] = sm_p
                sm_n_all[scheme][log_file[len("log_" + str(scheme) + "_") :]] = sm_n
                break

    print(f"{res_folder} has been loaded !!")
    return [
        time_all,
        bit_rate_all,
        buff_all,
        rebuf_all,
        bw_all,
        raw_reward_all,
        quality_all,
        sm_p_all,
        sm_n_all,
    ]


def get_performance(raw, scm_key, baseline, ini=4, end=49):
    scm_key_ = scm_key
    reward_all = {}
    reward_quality = {}
    reward_rebuf = {}
    reward_smooth_p = {}
    reward_smooth_n = {}
    reward_improvement = {}
    rebuf_improvement = {}

    time_all, _, _, rebuf_all, _, raw_reward_all, quality_all, sm_p_all, sm_n_all = (
        raw[0],
        raw[1],
        raw[2],
        raw[3],
        raw[4],
        raw[5],
        raw[6],
        raw[7],
        raw[8],
    )

    for scheme in scm_key_:
        reward_all[scheme] = []
        reward_quality[scheme] = []
        reward_rebuf[scheme] = []
        reward_smooth_p[scheme] = []
        reward_smooth_n[scheme] = []
        if scheme != baseline:
            reward_improvement[scheme] = []
            rebuf_improvement[scheme] = []

    for l in time_all[scm_key_[0]]:
        schemes_check = True
        for scheme in scm_key_:
            if l not in time_all[scheme] or len(time_all[scheme][l]) < end:
                schemes_check = False
                break
        if schemes_check:
            for scheme in scm_key_:
                # --------- record the total QoE data -----------
                reward_all[scheme].append(np.mean(raw_reward_all[scheme][l][ini:end]))
                # --------- record the individual terms in QoE --------
                # --------- chunk quality -----------
                trace_r_q = QUALITY_PENALTY * np.mean(quality_all[scheme][l][ini:end])
                reward_quality[scheme].append(trace_r_q)
                # -------- smoothness positive p ----------
                trace_sm_p = SMOOTH_PENALTY_P * np.mean(sm_p_all[scheme][l][ini:end])
                reward_smooth_p[scheme].append(trace_sm_p)
                # -------- smoothness positive n ----------
                trace_sm_n = SMOOTH_PENALTY_N * np.mean(sm_n_all[scheme][l][ini:end])
                reward_smooth_n[scheme].append(trace_sm_n)
                # -------- rebuffering penalty ------------
                trace_r = REBUF_PENALTY * np.mean(rebuf_all[scheme][l][ini:end])
                reward_rebuf[scheme].append(trace_r)

    # -------- calculate the reward improvement --------
    comparison_schemes = scm_key_.copy()
    comparison_schemes.remove(baseline)
    for l in range(len(reward_all[baseline])):
        for scheme in comparison_schemes:
            reward_improvement[scheme].append(
                -float((reward_all[baseline][l] - reward_all[scheme][l]) / 1.0)
            )  # abs(reward_all[scheme][l])
            rebuf_improvement[scheme].append(
                float(reward_rebuf[baseline][l] - reward_rebuf[scheme][l])
            )
    return (
        reward_all,
        reward_quality,
        reward_rebuf,
        reward_smooth_p,
        reward_smooth_n,
        reward_improvement,
        rebuf_improvement,
    )


def get_avg_terms(data, scm_key, scm_lbl, playback_d=4):
    """
    calculate the average QoE and individual terms (mean value + std)
    """
    mean_rewards = {}
    avg_terms = []
    r_all = data[0]
    r_q = data[1]
    r_rebuff = data[2]
    r_sm_p = data[3]
    r_sm_n = data[4]

    gap_flag = False
    for scm in reversed(scm_key):
        # for scm in scm_key:
        mean_rewards[scm] = np.mean(r_all[scm])
        info = []
        info.append(scm_lbl[scm])
        # ---------mean value and std -----------
        info.append(np.mean(r_all[scm]))
        info.append(np.std(r_all[scm]))
        # --------- Gap --------
        if not gap_flag:
            gap_flag = True
            opt_value = np.mean(r_all[scm])
            info.append(0)
        else:
            info.append(float(np.mean(r_all[scm]) - opt_value) / opt_value)
        # --------- quality ----------
        value = [float(e / (QUALITY_PENALTY * 100.0)) for e in r_q[scm]]
        info.append(np.mean(value))
        info.append(np.std(value) / 10)
        # --------- rebuffer penalty -----------
        value = [float(e / (playback_d * REBUF_PENALTY) * 100) for e in r_rebuff[scm]]
        # value = [float(e) for e in r_rebuff[scm]]
        info.append(np.mean(value))
        info.append(np.std(value) / 10)
        # --------- smoothness penalty -----------
        info.append(np.mean(r_sm_p[scm]))
        info.append(np.std(r_sm_p[scm]))
        info.append(np.mean(r_sm_n[scm]))
        info.append(np.std(r_sm_n[scm]))

        avg_terms.append(info)

    avg_terms = avg_terms[::-1]

    df = pd.DataFrame(
        avg_terms,
        columns=[
            "scheme",
            "reward_mean",
            "reward_std",
            "Gap",
            "quality_mean",
            "quality_std",
            "rebuff_mean",
            "rebuff_std",
            "smooth_p_mean",
            "smooth_p_std",
            "smooth_n_mean",
            "smooth_n_std",
        ],
    )
    return df, mean_rewards
