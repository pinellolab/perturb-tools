from typing import Dict
import pandas as pd
from ..._framework._ScreenModule import _Screen


def check_indices(df, index=None, colnames=None):
    if index is not None and not (df.index == index).all():
        raise ValueError("df.index does not match index, check your input.")
    if colnames is not None and not (df.columns == colnames).all():
        raise ValueError("df.columns does not match colnames, check your input.")


def read_csvs(
    X_filename: str,
    guides_filename: str = None,
    samples_filename: str = None,
    layers_filenames_dict: Dict[str, str] = None,
    matched_indices: bool = True,
) -> _Screen:
    """Returns Screen object with X, guides, and samples.
    Assumes first column is index and first row is colnames.
    matched_indies: If True, assumes X_df.columns == samples_df.index and X_df.index == guides_df.index
    """
    X_df = pd.read_csv(X_filename, index_col=0)
    if guides_filename:
        guides_df = pd.read_csv(guides_filename, index_col=0)
        if X_df.shape[0] != len(guides_df):
            raise ValueError(
                f"X has length {X_df.shape[0]} that does not match len(guides_df) {len(guides_df)}."
            )
    else:
        guides_df = None
    if samples_filename:
        samples_df = pd.read_csv(samples_filename, index_col=0)
        if X_df.shape[1] != len(samples_df):
            raise ValueError(
                f"X has width {X_df.shape[1]} that does not match len(samples_df) {len(samples_df)}."
            )
    else:
        samples_df = None

    if layers_filenames_dict:
        layers_dict = {}
        for key, df_path in layers_filenames_dict.items():
            layer_df = pd.read_csv(df_path, index_col=0)
    else:
        layers_dict = None

    if matched_indices:
        try:
            check_indices(X_df, guides_df.index, samples_df.index)
        except ValueError as e:
            raise ValueError("Error raised for X_df") from e
        if layers_filenames_dict:
            layers_dict = {}
            for key, df_path in layers_filenames_dict.items():
                try:
                    check_indices(layer_df, guides_df.index, samples_df.index)
                except ValueError as e:
                    raise ValueError("Error raised for layer_df") from e
                layers_dict[key] = layer_df
    else:
        if guides_df is not None:
            X_df = X_df.loc[guides_df.index, :]
            if layers_filenames_dict:
                for key, df_path in layers_filenames_dict.items():
                    layer_df = layer_df.loc[guides_df.index, :]
                    layers_dict[key] = layer_df
        if samples_df is not None:
            X_df = X_df.loc[:, samples_df.index]
            if layers_filenames_dict:
                for key, df_path in layers_filenames_dict.items():
                    layer_df = layer_df.loc[:, samples_df.index]
                    layers_dict[key] = layer_df
    if layers_dict:
        for key, df in layers_dict.items():
            layer_df = df.values

    return _Screen(
        X=X_df.values, guides=guides_df, samples=samples_df, layers=layers_dict
    )
