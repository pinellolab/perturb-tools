import numpy as np


def _read_count_normalize(X, pseudocount: int = 1):
    """Read depth normalization by sample. Assumes samples are columns and guides are rows."""

    return (X / np.nansum(X+pseudocount, axis=0)) * 1e6


def _log_transform_read_count(X, pseudocount: int = 1):
    """"""
    return np.log2(X + pseudocount)


def _log_normalize_read_count(X, pseudocount:int = 1):
    """Following the protocol written clearly, here:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0170445#sec002
    (see Methods).
    """

    X_read_norm = _read_count_normalize(X, pseudocount)
    X_log_read_norm = _log_transform_read_count(X_read_norm)

    return X_log_read_norm
