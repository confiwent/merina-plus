import math
import networkx as nx
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import pdb

warnings.filterwarnings(
    "ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning
)

COLOR_MAP = sns.color_palette(as_cmap=True)  # plt.cm.magma # viridis, jet n_colors = 8,
N_R = 4
LINE_STY = ["-", ":", "-.", "--", ":", "-.", ":", "-"]
POINT_STY = ["o", "s", "^", "*", ">", "v", "D", "s", "p"]

# sns.set_style("white")
# plt.style.use(['science','ieee'])
plt.style.use(["seaborn-paper"])
# sns.set(font_scale=1.5, font='Arial')
# plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.family"] = "Times New Roman"

STYLE = ["seaborn-white"]

options = {"node_size": 2, "edge_color": "black", "linewidths": 1, "width": 0.5}


def rotate_list_right(lst, n):
    n = n % len(lst)
    return lst[-n:] + lst[:-n]


def split_dict(d, c):
    sub_dicts = []
    sub_dict = {}
    for i, (k, v) in enumerate(d.items()):
        sub_dict[k] = v
        if (i + 1) % c == 0:
            sub_dicts.append(sub_dict)
            sub_dict = {}
    if sub_dict:
        sub_dicts.append(sub_dict)
    return sub_dicts


def plot_gaps(allgaps, title="title", max_num=32, thr=0, save_dir=None):
    # fig_num = math.ceil(len(allgaps)/max_num)
    data_list = split_dict(allgaps, max_num)
    for _ in range(len(data_list)):
        gaps = data_list[_]
        plt.figure(figsize=(10, 15))
        colors = ["C2" if val > thr else "C0" for val in gaps.values()]
        sns.barplot(y=list(gaps.keys()), x=list(gaps.values()), palette=colors)
        title_ = title + "_" + str(_)
        plt.title(title_)
        plt.xlabel("gaps", fontweight="bold")
        plt.ylabel("instance", fontweight="bold")
        save_fig(save_dir=save_dir, title=title_)


def plot_adj_comparison(graphs_gen, graphs_ori, title="tm", save_dir=None):
    compar_size = 10
    img_c = 2
    batch = np.random.choice(len(graphs_gen), compar_size, replace=False)
    title_str = ["ori", "gen"]
    save_dir_ = os.path.join(*[save_dir, "comparison"])
    fig_dir = os.path.join(*["samples", "fig", save_dir_])
    os.system("rm " + fig_dir + "/*")
    for idx in batch:
        figure = plt.figure()
        title_ = title + "_" + str(idx)
        if not isinstance(graphs_gen[idx], nx.Graph):
            G_gen = graphs_gen[idx].g.copy()
        else:
            G_gen = graphs_gen[idx].copy()
        if not isinstance(graphs_ori[idx], nx.Graph):
            G_ori = graphs_ori[idx].g.copy()
        else:
            G_ori = graphs_ori[idx].copy()
        assert isinstance(G_gen, nx.Graph)
        assert isinstance(G_ori, nx.Graph)

        # ------- generate adjacency ----------
        # node_list_gen = []
        # node_list_ori = []
        node_list = []
        for v, feature in G_gen.nodes.data("feature"):
            node_list.append(v)
            assert G_ori.nodes[v]["feature"].all() == feature.all()

        adj_gen = nx.to_numpy_matrix(G_gen, nodelist=node_list)
        adj_ori = nx.to_numpy_matrix(G_ori, nodelist=node_list)  # return weighted adj
        # adj_ori = np.where(adj_ori > 0, 1, 0)
        adjs = [adj_ori, adj_gen]
        for i in range(img_c):
            ax = plt.subplot(2, img_c, i + 1)
            sns.heatmap(adjs[i])
            ax.title.set_text(title_str[i])
        figure.suptitle(title_)
        save_fig(save_dir=save_dir_, title=title_)


def plot_graphs_list(graphs, title="title", max_num=16, save_dir=None, N=0):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    for i in range(max_num):
        # idx = i * (batch_size // max_num)
        idx = i + max_num * N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f"e={e - l}, n={v}"
        # if 'lobster' in save_dir.split('/')[0]:
        #     if is_lobster_graph(graphs[idx]):
        #         title_str += f' [L]'
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def save_fig(save_dir=None, title="fig", dpi=400):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(*[save_dir, "fig"])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(
            os.path.join(fig_dir, title),
            bbox_inches="tight",
            dpi=dpi,
            transparent=False,
        )
        plt.close()
    return


def save_pdf(save_dir=None, title="fig", dpi=400):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(*[save_dir, "fig"])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        # clear_folder(fig_dir)
        plt.savefig(
            os.path.join(fig_dir, title) + ".pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=dpi,
            format="pdf",
            transparent=False,
        )
        plt.close()
    return


def save_graph_list(log_folder_name, exp_name, gen_graph_list):
    if not (os.path.isdir("./samples/pkl/{}".format(log_folder_name))):
        os.makedirs(os.path.join("./samples/pkl/{}".format(log_folder_name)))
    with open("./samples/pkl/{}/{}.pkl".format(log_folder_name, exp_name), "wb") as f:
        pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = "./samples/pkl/{}/{}.pkl".format(log_folder_name, exp_name)
    return save_dir


def plot_scatter(
    data,
    save_dir,
    title="QoE_terms",
    label_x="Time spent stalled (%) ",
    label_y="Average qualities (dB)",
    tr_name="fcc",
):
    data_df = data  # dataframe
    x = data_df["rebuff_mean"].tolist()
    y = data_df["quality_mean"].tolist()
    legend = data_df["scheme"].tolist()
    x_err = data_df["rebuff_std"].tolist()
    y_err = data_df["quality_std"].tolist()
    # sns.scatterplot(x='rebuff_mean', y='quality_mean', hue= 'scheme', data=data_df)
    # sns.lineplot(x=x, y=y, color='b', alpha=0.7, linewidth=2)
    # with plt.style.context(['science', 'ieee']):
    # sns.set_context('paper')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_map = rotate_list_right(COLOR_MAP, N_R)
    colors = [color_map[i] for i in range(len(x))]
    fmts = [POINT_STY[i] for i in range(len(x))]
    for idx in range(len(x)):
        # ax.scatter(x=[x[idx]], y=[y[idx]],
        #             color = colors[idx], marker = fmts[idx], alpha=1, s=70, label=legend[idx])
        ax.errorbar(
            x[idx],
            y[idx],
            y_err[idx],
            x_err[idx],
            fmt=fmts[idx],
            color=colors[idx],
            capsize=3,
            elinewidth=1.6,
            capthick=1.6,
            label=legend[idx],
        )  # capsize=3

    legend_ = ax.legend(legend, loc="best", fontsize=16, ncol=1)
    frame = legend_.get_frame()
    frame.set_alpha(0)
    frame.set_facecolor("none")

    plt.ylabel(label_y, fontsize=20)
    plt.xlabel(label_x, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines["top"].set_linewidth(2.5)
    ax.spines["right"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)
    # plt.show()
    # save_fig(save_dir, title)
    title_ = title + "_" + tr_name
    save_pdf(save_dir, title_)


def plot_traces_comparison(
    data,
    means,
    scm_lbl,
    save_dir,
    title="traces_QoE",
    label_x="Session index",
    label_y="Avg. QoE",
    tr_name="fcc",
):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    SCHEMES_REW = []
    idx_min = 40
    idx_max = 90
    for _ in scm_lbl:
        ax.plot(data[_][idx_min:idx_max])
        # pdb.set_trace()
        # ax.scatter(x = list(range(idx_min, idx_max)), y=data[_][idx_min:idx_max],alpha=1, s=70)
        # SCHEMES_REW.append(scm_lbl[_] + ': ' + str('%.3f' % means[_]))
        SCHEMES_REW.append(scm_lbl[_])

    # color_map = rotate_list_right(COLOR_MAP, N_R)
    # colors = [color_map[i] for i in range(len(scm_lbl))]
    # for i,j in enumerate(ax.lines):
    #     j.set_color(colors[i])
    color_map = rotate_list_right(COLOR_MAP, N_R)
    colors = [color_map[i] for i in range(len(scm_lbl))]
    for i, j in enumerate(ax.lines):
        plt.setp(j, color=colors[i], linestyle=LINE_STY[i], linewidth=2.5)
    legend = ax.legend(SCHEMES_REW, loc="upper left", fontsize=14, ncol=2)
    # frame = legend.get_frame()
    # frame.set_alpha(0)
    # frame.set_facecolor('none')

    plt.ylabel(label_y, fontsize=20)
    plt.xlabel(label_x, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines["top"].set_linewidth(2.5)
    ax.spines["right"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)
    # plt.show()
    title_ = title + "_" + tr_name
    save_pdf(save_dir, title_)


def plot_cdf_traces(
    data, scm_lbl, save_dir, title="cdf", num_bins=200, xlim=None, tr_name="fcc"
):
    SCHEMES_REW = []
    # with plt.style.context(STYLE):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for scheme in scm_lbl:
        SCHEMES_REW.append(scm_lbl[scheme])
        values, base = np.histogram(data[scheme], bins=num_bins)
        cumulative = np.cumsum(values) / float(len(data[scheme]))
        ax.plot(base[:-1], cumulative)

    color_map = rotate_list_right(COLOR_MAP, N_R)
    colors = [color_map[i] for i in range(len(scm_lbl))]
    for i, j in enumerate(ax.lines):
        plt.setp(j, color=colors[i], linestyle=LINE_STY[i], linewidth=2.6)

    legend = ax.legend(SCHEMES_REW, loc="best", fontsize=18)
    frame = legend.get_frame()
    frame.set_alpha(0)
    frame.set_facecolor("none")

    plt.ylabel("CDF (Perc. of sessions)", fontsize=20)
    plt.xlabel("Average values of chunk's QoE", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.ylim([-0.02, 1.02])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)
    # plt.title(title)
    # plt.show()
    # save_fig(save_dir=save_dir, title=title)
    title_ = title + "_" + tr_name
    save_pdf(save_dir, title_)


def plot_cdf_improvement(
    data,
    scm_lbl,
    cp_lbl,
    save_dir,
    title="cdf_im",
    num_bins=200,
    xlim=None,
    tr_name="fcc",
):
    SCHEMES_REW = []
    # with plt.style.context(STYLE):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for scheme in cp_lbl:
        SCHEMES_REW.append(scm_lbl[scheme])
        values, base = np.histogram(data[scheme], bins=num_bins)
        cumulative = np.cumsum(values) / float(len(data[scheme]))
        ax.plot(base[:-1], cumulative)

    color_map = rotate_list_right(COLOR_MAP, N_R)
    colors = [color_map[i] for i in range(len(scm_lbl))]
    for i, j in enumerate(ax.lines):
        plt.setp(
            j,
            color=colors[i + 1] if i > 0 else colors[i],
            linestyle=LINE_STY[i],
            linewidth=2.5,
        )

    legend = ax.legend(SCHEMES_REW, loc="best", fontsize=18)
    frame = legend.get_frame()
    frame.set_alpha(0)
    frame.set_facecolor("none")

    plt.ylabel("CDF (Perc. of sessions)", fontsize=20)
    plt.xlabel("Average QoE improvement", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.ylim([-0.02, 1.02])
    plt.vlines(0, 0, 1, colors=colors[1], linestyles="solid", linewidths=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)
    # plt.title(title)
    # plt.show()
    # save_fig(save_dir=save_dir, title=title)
    title_ = title + "_" + tr_name
    save_pdf(save_dir, title_)


def plot_curves(
    data_mean,
    data_er,
    data_ep,
    lables,
    save_dir,
    title="adapt_curves",
    label_x="Training Epochs",
    label_y="Avg. QoE",
    tr_name="fcc",
):
    y_mu = data_mean
    y_err = data_er
    x = data_ep
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_map = COLOR_MAP
    colors = [color_map[i] for i in range(len(x))]
    fmts = [POINT_STY[i] for i in range(len(x))]
    lmts = [LINE_STY[i] for i in range(len(x))]
    id = 0
    end = int(44)
    for idx in x:
        # ax.scatter(x=[x[idx]], y=[y[idx]],
        #             color = colors[idx], marker = fmts[idx], alpha=1, s=70, label=legend[idx])
        # ax.errorbar(x[idx][:end], y_mu[idx][:end], y_err[idx][:end], \
        #                 fmt=fmts[id], color = colors[id], \
        #                 capsize=3, elinewidth= 1.6,
        #                 capthick = 1.6, label=lables[id]) # capsize=3
        x_ = []
        y_mu_ = []
        y_err_ = []
        for _ in range(end):
            if _ % 2 == 0:
                x_.append(x[idx][_])
                y_mu_.append(y_mu[idx][_])
                y_err_.append(y_err[idx][_])
        # ax.plot(x[idx][:end], y_mu[idx][:end], alpha=0.9)
        # ax.errorbar(x_, y_mu_, y_err_, \
        #                 fmt=fmts[id], color = colors[id], \
        #                 capsize=3, elinewidth= 1.6,
        #                 capthick = 1.6, label=lables[id]) # capsize=3
        ax.plot(
            x_,
            y_mu_,
            marker=fmts[id],
            color=colors[id],
            alpha=0.8,
            linewidth=2.5,
            markersize=10,
        )
        id += 1
    # for i,j in enumerate(ax.lines):
    #     plt.setp(j, color = colors[i], linestyle = lmts[i], linewidth = 2.5)

    legend_ = ax.legend(lables, loc="best", fontsize=16, ncol=1)
    frame = legend_.get_frame()
    frame.set_alpha(0)
    frame.set_facecolor("none")

    plt.ylabel(label_y, fontsize=20)
    plt.xlabel(label_x, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(left=0)
    ax.spines["top"].set_linewidth(2.5)
    ax.spines["right"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.spines["left"].set_linewidth(2.5)
    # plt.show()
    # save_fig(save_dir, title)
    title_ = title + "_" + tr_name
    save_pdf(save_dir, title_)
