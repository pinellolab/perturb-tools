import numpy as np
import numpy.ma as ma
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

linestyles = ["solid", "dotted", "dashed", "dashdot"]
corr_func_dict = {
    "Pearson": lambda X: ma.corrcoef(X.T),
    "Spearman": lambda X: scipy.stats.spearmanr(X, nan_policy="omit")[0],
}


def set_count_stats(screen):
    screen.condit["median_X"] = np.nanmedian(screen.X, axis=0)
    screen.condit["median_X_bcmatch"] = np.nanmedian(screen.layers["X_bcmatch"], axis=0)
    if "edits" in screen.layers.keys():
        screen.condit["median_edit"] = np.nanmedian(screen.layers["edits"], axis=0)
        screen.get_guide_edit_rate()
        screen.condit["median_edit_rate"] = np.nanmedian(
            screen.layers["edit_rate"], axis=0
        )


def set_sample_correlation_guides(screen, guide_idx, prefix="", method="Pearson"):
    corr_func = corr_func_dict[method]
    screen_subset = screen[guide_idx, :].copy()
    if prefix != "":
        prefix = f"{prefix}_"
    c = corr_func(ma.masked_invalid(screen_subset.X))
    if c is np.isnan:
        print(
            f"Failed to calculate {method} correlation. Check if your matrix is constant or have no valid values."
        )
        return
    screen.varm[f"{prefix}corr_X"] = c

    # screen.condit[f"{prefix}mean_corr_X"] = screen.varm[f"{prefix}corr_X"].mean(0)

    screen.condit[f"{prefix}median_corr_X"] = np.median(
        screen.varm[f"{prefix}corr_X"], axis=0
    )


def set_sample_correlation(screen, method="Pearson"):
    set_sample_correlation_guides(screen, screen.guides.index, prefix="", method=method)


def plot_correlation(screen, method="Pearson"):
    """
    arguments
    -- method: ["Pearson", "Spearman"]
    """
    set_sample_correlation(screen, method)
    n_corrs = len(screen.varm.keys())

    def plot_heatmap(k, cor, ax):
        sns.heatmap(
            cor,
            ax=ax,
            xticklabels=screen.condit.index,
            yticklabels=screen.condit.index,
        )
        ax.set_title(k)
        ax.set_aspect(1)

    fig, ax = plt.subplots(n_corrs, 1, figsize=(10, n_corrs * 10))
    for i, (k, cor) in enumerate(screen.varm.items()):
        if n_corrs > 1:
            plot_heatmap(k, cor, ax[i])
        else:
            plot_heatmap(k, cor, ax)
            break
    return ax


def plot_lfc_correlation(
    screen,
    guide_idx,
    lfc_label_suffix="",
    method="Pearson",
    cond1="top",
    cond2="bot",
    rep_condit="replicate",
    figsize=(7, 7),
    **kwargs,
):
    screen_subset = screen[guide_idx, :]
    lfcs = screen_subset.log_fold_change_reps(
        cond1=cond1, cond2=cond2, rep_condit=rep_condit, **kwargs
    )
    screen.uns["lfc"] = lfcs
    screen.uns["lfc_corr"] = corr_func_dict[method](lfcs)
    for i, col in enumerate(lfcs.columns):
        rep = col.split(".")[0]
        screen.condit.loc[
            screen.condit[rep_condit] == rep, f"median_lfc_corr.{cond1}_{cond2}"
        ] = np.nanmedian(screen.uns["lfc_corr"][i, :])
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        screen.uns["lfc_corr"],
        ax=ax,
        annot=True,
        xticklabels=lfcs.columns.map(lambda s: s.split(".")[0]),
        yticklabels=lfcs.columns.map(lambda s: s.split(".")[0]),
    )
    ax.set_aspect(1)
    return ax


def G(v):
    bins = np.linspace(0.0, 1.0, 50)
    total = float(np.sum(v))
    yvals = []
    for b in bins:
        bin_vals = v[v <= np.quantile(v, b)]
        bin_fraction = np.sum(bin_vals) / total
        yvals.append(bin_fraction)
    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)
    return bins, yvals, gini_val


def plot_guide_coverage(screen, ax=None, figsize=(4, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for i, c in enumerate(screen.condit.index):
        sns.kdeplot(
            screen.X[:, i] + 1,
            label=f"{c} (median {np.nanmedian(screen.X[:, i])})",
            clip=(1, screen.X.max() + 10),
            linestyle=linestyles[(i // 10) % 4],
            ax=ax,
        )
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xscale("log")
    ax.set_xlabel("counts")
    ax.set_ylabel("# guides")
    ax.set_title("Guide coverage")
    return ax


def plot_X_gini(screen, plot=True):
    gini_vals = []
    if plot:
        plt.figure()
    for c in range(len(screen.condit.index)):
        bins, result, gini_val = G(screen.X[:, c])
        gini_vals.append(gini_val)
        if plot:
            plt.plot(
                bins,
                result,
                label="{} ({:.3f})".format(screen.condit.index[c], gini_val),
            )
    if plot:
        plt.plot(bins, bins, "--", label="perfect eq.")
        plt.xlabel("Cumulative fraction of guides \nfrom lowest to highest read counts")
        plt.ylabel("Cumulative fraction of read counts")
        plt.title("GINI index")
        plt.legend(bbox_to_anchor=(1.02, 1))
    screen.condit["gini_X"] = gini_vals


def get_outlier_guides(
    screen, condit_col: str, mad_z_thres: float = 5, abs_RPM_thres: float = 10000
):
    """Obtain outlier guides.
    For each experimental condition in `condit`, find outlier guides that shows extreme counts compared to guides.
    Outlier guides are defined as those With MAD criteria (z>`mad_z_thres`) and has absolute RPM of `abs_RPM_thres` are defined as outlier guides.
    """

    if "X_RPM" not in screen.layers:
        screen.layers["X_RPM"] = screen.X / screen.X.sum(axis=0) * 10e6
    aberr_dict = {}

    for cond in screen.condit[condit_col].unique():
        adata = screen[:, screen.condit[condit_col] == cond]
        median_p = np.nanmedian(adata.layers["X_RPM"], axis=1)
        aberr_guide_df_condit = []
        aberr_idx_list = []
        for i, sample in enumerate(adata.condit.index):
            outlier_idx = np.where(
                (adata.layers["X_RPM"][:, i] > median_p * mad_z_thres)
                & (adata.layers["X_RPM"][:, i] > abs_RPM_thres)
            )[0]
            outlier_guides = adata.guides.iloc[outlier_idx, :].copy()
            outlier_guides["sample"] = adata.condit.index[i]
            outlier_guides["RPM"] = adata.layers["X_RPM"][outlier_idx, i]
            aberr_guide_df_condit.append(outlier_guides[["sample", "RPM"]])
        aberr_dict[cond] = aberr_guide_df_condit
    aberr_guide_dfs = []
    for df in aberr_dict.values():
        aberr_guide_dfs.extend(df)
    aberr_guides = pd.concat(aberr_guide_dfs, axis=0)
    # aberr_guides.index = aberr_idx_list
    return aberr_guides.reset_index()
