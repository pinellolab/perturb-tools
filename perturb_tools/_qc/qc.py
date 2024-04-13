import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

linestyles = ["solid", "dotted", "dashed", "dashdot"]


def spearman_keepdim(X):
    # print(scipy.stats.spearmanr(X, nan_policy="omit"))
    # c = scipy.stats.spearmanr(X, nan_policy="omit")[0]
    df = pd.DataFrame(X)
    c = df.corr(method="spearman").values
    return c


corr_func_dict = {
    "Pearson": lambda X: ma.corrcoef(X.T),
    "Spearman": lambda X: spearman_keepdim(X),
}


def set_count_stats(screen):
    screen.samples["median_X"] = np.nanmedian(screen.X, axis=0)
    screen.samples["median_X_bcmatch"] = np.nanmedian(
        screen.layers["X_bcmatch"], axis=0
    )
    if "edits" in screen.layers.keys():
        screen.samples["median_edit"] = np.nanmedian(screen.layers["edits"], axis=0)
        screen.get_guide_edit_rate()
        screen.samples["median_edit_rate"] = np.nanmedian(
            screen.layers["edit_rate"], axis=0
        )


def set_sample_correlation_guides(screen, guide_idx, prefix="", method="Pearson"):
    corr_func = corr_func_dict[method]
    screen_subset = screen[guide_idx, :].copy()
    if prefix != "":
        prefix = f"{prefix}_"
    c = corr_func(ma.masked_invalid(screen_subset.X))
    if np.isnan(c).all():
        print(
            f"Failed to calculate {method} correlation. Check if your matrix is constant or have no valid values."
        )
        return
    screen.varm[f"{prefix}corr_X"] = c

    # screen.samples[f"{prefix}mean_corr_X"] = screen.varm[f"{prefix}corr_X"].mean(0)

    screen.samples[f"{prefix}median_corr_X"] = np.nanmedian(
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
            xticklabels=screen.samples.index,
            yticklabels=screen.samples.index,
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
    rep_col="replicate",
    figsize=(7, 7),
    **kwargs,
):
    screen_subset = screen[guide_idx, :]
    lfcs = screen_subset.log_fold_change_reps(
        cond1=cond1, cond2=cond2, rep_col=rep_col, **kwargs
    )
    screen.uns["lfc"] = lfcs
    screen.uns["lfc_corr"] = corr_func_dict[method](lfcs)
    for i, col in enumerate(lfcs.columns):
        rep = col.split(f".{cond1}_{cond2}", 1)[0]
        if "." in rep:
            rep = rep.split(".")
        if isinstance(rep_col, str):
            rep_idx = screen.samples[rep_col].astype(str) == rep
        else:
            rep_idx = (screen.samples[rep_col] == rep).all(axis=1)
        screen.samples.loc[
            rep_idx,
            f"median_lfc_corr.{cond1}_{cond2}",
        ] = np.nanmedian(np.delete(screen.uns["lfc_corr"][i, :], i))
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
    for i, c in enumerate(screen.samples.index):
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
    for c in range(len(screen.samples.index)):
        bins, result, gini_val = G(screen.X[:, c])
        gini_vals.append(gini_val)
        if plot:
            plt.plot(
                bins,
                result,
                label="{} ({:.3f})".format(screen.samples.index[c], gini_val),
            )
    if plot:
        plt.plot(bins, bins, "--", label="perfect eq.")
        plt.xlabel("Cumulative fraction of guides \nfrom lowest to highest read counts")
        plt.ylabel("Cumulative fraction of read counts")
        plt.title("GINI index")
        plt.legend(bbox_to_anchor=(1.02, 1))
    screen.samples["gini_X"] = gini_vals


def get_outlier_guides(
    screen, condit_col: str, mad_z_thres: float = 5, abs_RPM_thres: float = 10000
):
    """Obtain outlier guides.
    For each experimental condition in `samples`, find outlier guides that shows extreme counts compared to guides.
    Outlier guides are defined as those With MAD criteria (z>`mad_z_thres`) and has absolute RPM of `abs_RPM_thres`.
    """

    if "X_RPM" not in screen.layers:
        screen.layers["X_RPM"] = (screen.X / screen.X.sum(axis=0) * 10e6).copy()
    aberr_dict = {}

    for cond in screen.samples[condit_col].unique():
        adata = screen[:, screen.samples[condit_col] == cond]
        median_p = np.nanmedian(adata.layers["X_RPM"].copy(), axis=1)
        aberr_guide_df_condit = []
        for i, sample in enumerate(adata.samples.index):
            outlier_idx = np.where(
                (adata.layers["X_RPM"][:, i] > median_p * mad_z_thres)
                & (adata.layers["X_RPM"][:, i] > abs_RPM_thres)
            )[0]
            outlier_guides = adata.guides.iloc[outlier_idx, :].copy()
            outlier_guides["sample"] = adata.samples.index[i]
            outlier_guides["RPM"] = adata.layers["X_RPM"].copy()[outlier_idx, i]
            aberr_guide_df_condit.append(outlier_guides[["sample", "RPM"]])
        aberr_dict[cond] = aberr_guide_df_condit
    aberr_guide_dfs = []
    for df in aberr_dict.values():
        aberr_guide_dfs.extend(df)
    if len(aberr_guide_dfs) == 0:
        return pd.DataFrame({"name": [], "sample": [], "RPM": []})
    aberr_guides = pd.concat(aberr_guide_dfs, axis=0).reset_index()
    aberr_guides.columns = ["name"] + aberr_guides.columns[1:].tolist()
    # aberr_guides.index = aberr_idx_list
    return aberr_guides
