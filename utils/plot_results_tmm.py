import os, argparse, sys, pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_plot import check_folder, save_csv, save_excel
from loader_plot import (
    load_source_settings,
    load_labels,
    load_data_from_log,
    get_performance,
    get_avg_terms,
)
from plot import (
    plot_traces_comparison,
    plot_cdf_traces,
    plot_scatter,
    plot_cdf_improvement,
)

sys.path.append("./")
from config import args_plot

PIC_FOLDER = "./Results/pic/"
RES_FOLDER = "./Results/test/"
LABEL_FILE = "./config/labels.json"
DATA_FILE = "./config/data_source.json"
NUM_BINS = 200
BITS_IN_BYTE = 8.0
INIT_CHUNK = 4
M_IN_K = 1000.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
K_IN_M = 1000.0


class abr_datas:
    def __init__(self, data_path, save_path, scm_list) -> None:
        self.data_path = data_path  # include source data path and pic-saving path
        self.scm_list = scm_list
        self.raw_data = {}
        self.get_plot_bl()
        self.load_raw_data()
        self.save_path = save_path
        self.baseline = "test_mpc"
        self.video_ini = int(4)
        self.video_end = int(49)
        self.chunk_duration = 4.0
        self.xlim = None

    def get_plot_bl(self):
        key = []
        bl = {}
        for scm in self.scm_list:
            key.append(scm[0])
            bl[scm[0]] = scm[1]
        self.scm_key = key
        self.scm_lbl = bl

    def load_raw_data(self):
        self.raw_data = load_data_from_log(self.scm_key, self.data_path)

    def get_baseline(self, args):
        for scm in self.scm_key:
            # pdb.set_trace()
            if scm == args.baseline:
                self.baseline = scm

    def cal_performance(self):
        (
            self.r_all,
            self.r_quality,
            self.r_rebuf,
            self.r_smooth_p,
            self.r_smooth_n,
            self.r_imp,
            self.rebuf_imp,
        ) = get_performance(
            self.raw_data, self.scm_key, self.baseline, self.video_ini, self.video_end
        )

    def cal_avg_terms(self):
        data = [
            self.r_all,
            self.r_quality,
            self.r_rebuf,
            self.r_smooth_p,
            self.r_smooth_n,
        ]
        self.avg_terms_df, self.mean_r = get_avg_terms(
            data, self.scm_key, self.scm_lbl, self.chunk_duration
        )
        save_path_ = os.path.join(*[self.save_path, "results.xlsx"])
        save_excel(self.avg_terms_df, save_path_)

    def get_results(self):
        self.cal_performance()
        self.cal_avg_terms()

    def plot_results(self, args):
        fig_dir = os.path.join(*[self.save_path, "fig"])
        cmd = "rm -r " + fig_dir
        os.system(cmd)
        plot_traces_comparison(
            self.r_all, self.mean_r, self.scm_lbl, self.save_path, tr_name=args.trace
        )
        if args.xlim_min != -99:
            self.xlim = [args.xlim_min, args.xlim_max]
        plot_cdf_traces(
            self.r_all, self.scm_lbl, self.save_path, xlim=self.xlim, tr_name=args.trace
        )
        plot_scatter(self.avg_terms_df, self.save_path, tr_name=args.trace)
        if args.xlim_min != -99:
            self.xlim = [args.xlim_min, args.xlim_max_im]
        comparison_schemes = self.scm_key.copy()
        comparison_schemes.remove(self.baseline)
        plot_cdf_improvement(
            self.r_imp,
            self.scm_lbl,
            comparison_schemes,
            self.save_path,
            xlim=self.xlim,
            tr_name=args.trace,
        )


def main():
    # -------- load arguments ----------
    parser = argparse.ArgumentParser()
    _, rest_args = parser.parse_known_args()
    args = args_plot.get_args(rest_args)

    # -------- load data settings ---------
    data_path = load_source_settings(args, RES_FOLDER, PIC_FOLDER, DATA_FILE)
    scm_list = load_labels(args, LABEL_FILE)

    check_folder(PIC_FOLDER)

    for _ in data_path:
        data_path_ = data_path[_][0]
        save_path_ = data_path[_][1]
        check_folder(save_path_)

        # -------- load data ---------
        agent = abr_datas(data_path_, save_path_, scm_list)
        agent.get_baseline(args)  # run before getting the performance
        agent.get_results()

        # -------- plot the results ----------
        if _ == "puffer3":
            args.xlim_min = -100
            args.xlim_max = 90
            args.xlim_max_im = 50
        elif _ == "puffer":
            args.xlim_min = -80
            args.xlim_max = 80
            args.xlim_max_im = 50
        else:
            args.xlim_min = -99
            args.xlim_max = -99
            args.xlim_max_im = -99
        args.trace = _
        if not args.nplot:
            agent.plot_results(args)


if __name__ == "__main__":
    main()
