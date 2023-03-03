import numpy as np
import numpy.ma as ma
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

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

    screen.condit[f"{prefix}mean_corr_X"] = screen.varm[f"{prefix}corr_X"].mean(0)

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
    screen, guide_idx, lfc_label_suffix="", method="Pearson", **kwargs
):
    screen_subset = screen[guide_idx, :]
    lfcs = screen_subset.log_fold_change_reps(**kwargs)
    screen.uns["lfc_corr"] = corr_func_dict[method](lfcs)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        screen.uns["lfc_corr"],
        ax=ax,
        annot=True,
        xticklabels=lfcs.map(lambda s: s.split(".")[0]),
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


def set_X_gini(screen, plot=True):
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
