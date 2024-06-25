# _ScreenModule.py
__module_name__ = "_ScreenModule.py"
__author__ = ", ".join(["Michael E. Vinyard", "Jayoung Kim Ryu"])
__email__ = ", ".join(["vinyard@g.harvard.edu", "jayoung_ryu@g.harvard.edu"])

import warnings
from typing import Union, List

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from .._arithmetic._funcs._log_fold_change import _log_fold_change
from .._normalization._funcs._read_count_norm import _log_normalize_read_count
from .._readwrite._funcs._read_screen_from_PoolQ import _read_screen_from_PoolQ
from .._readwrite._funcs._write_screen_to_csv import _write_screen_to_csv
from .._readwrite._funcs._write_screen_to_excel import _write_screen_to_excel
from .._utilities._funcs._update_dict import _update_dict
from ._supporting_functions._guides._GuideAnnotationModule import _annotate_sgRNAs
from ._supporting_functions._print_screen_object import _print_screen_object


class _Screen(AnnData):
    def __init__(self, X=None, guides=None, samples=None, *args, **kwargs):
        super().__init__(X=X, obs=guides, var=samples, *args, **kwargs)

    @classmethod
    def from_adata(cls, adata: ad.AnnData):
        return cls(
            (adata.X),
            guides=(adata.obs),
            samples=(adata.var),
            obsm=adata.obsm,
            obsp=adata.obsp,
            uns=(adata.uns),
            layers=(adata.layers),
        )

    @property
    def guides(self):
        return self.obs

    @guides.setter
    def guides(self, new_val: pd.DataFrame):
        self.obs = new_val

    @property
    def samples(self):
        return self.var

    @samples.setter
    def samples(self, new_val: pd.DataFrame):
        self.var = new_val

    @property
    def samples_m(self):
        return self.varm

    @samples_m.setter
    def samples_m(self, new_val):
        self.varm = new_val

    @property
    def samples_p(self):
        return self.varp

    @samples_p.setter
    def samples_p(self, new_val):
        self.varp = new_val

    def __repr__(self) -> str:
        return _print_screen_object(self)[2]

    def __add__(self, other):
        if all(self.guides.index == other.guides.index) and all(
            self.samples.index == other.samples.index
        ):
            return _Screen(self.X + other.X, self.guides.copy(), self.samples.copy())
        else:
            raise ValueError("Guides/sample description mismatch")

    def __getitem__(self, index):
        adata = super().__getitem__(index)
        return _Screen.from_adata(adata)

    def read_PoolQ(self, path, metadata=False, merge_metadata_on="Condition"):
        """Read poolQ."""
        self._PoolQ_outpath = path
        self._PoolQScreenDict = _read_screen_from_PoolQ(self._PoolQ_outpath)

        for key, value in _update_dict(self._PoolQScreenDict).items():
            self.__setattr__(key, value)

        if metadata:
            self.samples = self.samples.merge(
                pd.read_csv(metadata), on=merge_metadata_on
            )

        _print_screen_object(self)

    def annotate_guides(
        self, genes, chrom, start, stop, annotations, DirectPairDict, ref_seq_path
    ):
        """
        Annotate sgRNA table.

        """
        self.guides = _annotate_sgRNAs(
            self.guides,
            genes,
            chrom,
            start,
            stop,
            annotations,
            DirectPairDict,
            ref_seq_path,
        )

    def log_norm(self, output_layer="lognorm_counts", read_count_layer=None, pseudocount = 1):
        if read_count_layer is None:
            self.layers[output_layer] = _log_normalize_read_count(self.X, pseudocount)
        else:
            output_layer = f"lognorm_{read_count_layer}"
            self.layers[output_layer] = _log_normalize_read_count(
                self.layers[read_count_layer], pseudocount
            )

    # TBD: mask ones with too low raw counts.
    def log_fold_change(
        self,
        sample1,
        sample2,
        lognorm_counts_key="lognorm_counts",
        name=False,
        pseudocount: int = 1,
        out_guides_suffix="lfc",
        return_result=False,
    ):
        """
        General module to calculate LFC across experimental conditions.
        """
        if lognorm_counts_key == "lognorm_counts":
            self.log_norm(pseudocount=pseudocount)
        else:
            if "lognorm_" not in lognorm_counts_key:
                raise ValueError(f"{lognorm_counts_key} is not a lognorm layer- feed in 'lognorm_`layer_key`' as lognorm_counts_key.")
            read_count_layer_key = lognorm_counts_key.split("lognorm_")[-1]
            if read_count_layer_key not in self.layers:
                raise ValueError(f"{read_count_layer_key} not in .layers - feed in 'lognorm_`layer_key`' as lognorm_counts_key.")
            self.log_norm(output_layer=lognorm_counts_key, read_count_layer=read_count_layer_key, pseudocount=pseudocount)
        sample1_idx = np.where(sample1 == self.samples.index)[0]
        sample2_idx = np.where(sample2 == self.samples.index)[0]
        if len(sample1_idx) != 1 or len(sample2_idx) != 1:
            if len(sample1_idx) == 0:
                print(f"No sample named {sample1} in Screen object.")
            else:
                print(f"Duplicate sample name {sample1} in Screen object")
            if len(sample2_idx) == 0:
                print(f"No sample named {sample2} in Screen object.")
            else:
                print(f"Duplicate sample name {sample2} in Screen object")
            raise ValueError("")

        try:
            lfc = _log_fold_change(
                self.layers[lognorm_counts_key], sample1_idx, sample2_idx
            )
            if return_result:
                return lfc
            else:
                self.guides[f"{sample1}_{sample2}.{out_guides_suffix}"] = lfc
        except Exception:  # TBD: what error?
            print("Calculating LFC against two previously calculated LFC values...")
            dlfc = _log_fold_change(self.guides, sample1, sample2)

            if not name:
                name = f'{sample1.strip(".lfc")}_{sample2.strip(".lfc")}.d{out_guides_suffix}'

            if return_result:
                return dlfc
            else:
                self.guides[name] = dlfc

    def log_fold_change_reps(
        self,
        cond1,
        cond2,
        lognorm_counts_key="lognorm_counts",
        rep_col: Union[str, List[str]] = "replicate",
        compare_col="sort",
        out_guides_suffix="lfc",
        pseudocount=1,
        keep_result=False,
        ignore_missing=False,
    ):
        """Get gRNA abundance LFC across conditions for each replicate.
        ignore_missing: If True, does not raise Error when one of the conditions is missing for a replicate.
        """
        if isinstance(rep_col, str) and rep_col not in self.samples.columns:
            raise ValueError(f"{rep_col} not in samples features")
        elif isinstance(rep_col, list):
            for rc in rep_col:
                if rc not in self.samples.columns:
                    raise ValueError(f"{rc} not in samples features")
        if compare_col not in self.samples.columns:
            raise ValueError(f"{compare_col} not in samples features")
        if (
            cond1 not in self.samples[compare_col].tolist()
            or cond2 not in self.samples[compare_col].tolist()
        ):
            raise ValueError(
                f"{cond1} or {cond2} not in sample conditions {self.samples[compare_col]}"
            )

        lfcs = []

        if isinstance(rep_col, str):
            lfc_rep_cols = self.samples[rep_col].unique().tolist()
            unique_reps = self.samples[rep_col].unique()
        else:
            unique_reps = [
                list(t)
                for t in self.samples[rep_col].drop_duplicates().to_records(index=False)
            ]
            lfc_rep_cols = [
                ".".join(list(map(str, rep_list))) for rep_list in unique_reps
            ]
        for rep in unique_reps:
            if isinstance(rep_col, str):
                cond1_idx = np.where(
                    (self.samples[rep_col] == rep)
                    & (self.samples[compare_col] == cond1)
                )[0]
                cond2_idx = np.where(
                    (self.samples[rep_col] == rep)
                    & (self.samples[compare_col] == cond2)
                )[0]
            elif isinstance(rep_col, list):
                cond1_idx = np.where(
                    (self.samples[rep_col] == rep).all(axis=1)
                    & (self.samples[compare_col] == cond1)
                )[0]
                cond2_idx = np.where(
                    (self.samples[rep_col] == rep).all(axis=1)
                    & (self.samples[compare_col] == cond2)
                )[0]
            if len(cond1_idx) != 1 or len(cond2_idx) != 1:
                if not ignore_missing:
                    raise ValueError(
                        f"Conditions are not unique for each replicates ({rep_col} == {rep}) to be aggregated:\n{cond1}:{self.samples[cond1_idx]} or {cond2}:{self.samples[cond2_idx]}\n {self.samples}"
                    )
                else:
                    lfc_rep_cols.pop(rep)
                    continue

            lfcs.append(
                self.log_fold_change(
                    self.samples.index[cond1_idx].tolist()[0],
                    self.samples.index[cond2_idx].tolist()[0],
                    lognorm_counts_key=lognorm_counts_key,
                    pseudocount=pseudocount,
                    return_result=True,
                )
            )

        lfcs_array = np.concatenate(lfcs, axis=1)
        lfcs_df_columns = [
            f"{s}.{cond1}_{cond2}.{out_guides_suffix}" for s in lfc_rep_cols
        ]
        lfcs_df = pd.DataFrame(
            lfcs_array, index=self.guides.index, columns=lfcs_df_columns
        )

        if keep_result:
            self.guides[lfcs_df_columns] = lfcs_df
        return lfcs_df

    # TODO: add guides metadata on how aggregates are calcualted?
    def log_fold_change_agg(
        self,
        cond1,
        cond2,
        lognorm_counts_key="lognorm_counts",
        agg_col="replicate",
        compare_col="condition",
        out_guides_suffix="lfc",
        agg_fn="median",
        pseudocount=1,
        name=None,
        return_result=False,
        keep_per_replicate=False,
    ):
        lfcs_df = self.log_fold_change_reps(
            cond1,
            cond2,
            lognorm_counts_key=lognorm_counts_key,
            rep_col=agg_col,
            compare_col=compare_col,
            out_guides_suffix=out_guides_suffix,
            pseudocount=pseudocount,
            keep_result=keep_per_replicate,
        )

        if agg_fn == "mean":
            lfcs_agg = lfcs_df.apply(np.mean, axis=1)
        elif agg_fn == "median":
            lfcs_agg = lfcs_df.apply(np.median, axis=1)
        elif agg_fn == "sd":
            lfcs_agg = lfcs_df.apply(np.std, axis=1)
        else:
            raise ValueError(
                "Only 'mean', 'median', and 'sd' are supported for aggregating LFCs."
            )

        if return_result:
            return lfcs_agg
        if name is None:
            self.guides[f"{cond1}_{cond2}.{out_guides_suffix}.{agg_fn}"] = lfcs_agg
        else:
            self.guides[name] = lfcs_agg

    def fold_change(
        self,
        cond1: str,
        cond2: str,
        lognorm_counts_key: str = "lognorm_counts",
        return_result: bool = False,
    ) -> Union[None, pd.Series]:
        """Calculate log fold change (cond1/cond2) of normalized guide abundances."""

        log_fold_change = _log_fold_change(
            self.layers[lognorm_counts_key], cond1, cond2
        )
        if return_result:
            return log_fold_change
        self.guides[f"{cond1}_{cond2}.fc"] = log_fold_change

    def to_Excel(
        self,
        workbook_path: str = "CRISPR_screen.workbook.xlsx",
        index: bool = False,
        silent: bool = False,
        include_uns: bool = False,
    ) -> None:
        """Write components of Screen class to an Excel workbook.

        Args
        ---
        workbook_path: Prevent printing outpaths / details of the created workbook.
        index: If True, include an index in the workbook sheet.
        silent: If True, prevent printing outpaths / details of the created workbook.

        Notes:
        ------
        (1) Will likely need to be updated once we fully transition over to AnnData-like class.
        """

        _write_screen_to_excel(
            self,
            workbook_path,
            index,
            silent,
            include_uns,
        )

    def to_csv(self, out_path="CRISPR_screen"):
        """

        Write .csv files for each part of the screen. will eventually be replaced by something more native to AnnData.

        """

        _write_screen_to_csv(self, out_path)

    def to_mageck_input(
        self,
        out_path=None,
        count_layer=None,
        sgrna_column=None,
        target_column="target_id",
        sample_prefix="",
    ):
        """Formats screen object into mageck input.

        If screen.guides[target_column] is None or np.isnan, remove that guide
        """
        if count_layer is None:
            count_matrix = self.X
        else:
            try:
                count_matrix = self.layers[count_layer]
            except KeyError as exc:
                raise KeyError(
                    f"Layer {count_layer} doesn't exist in Screen object with layers {self.layers.keys()}"
                ) from exc
        mageck_input_df = (
            pd.DataFrame(
                count_matrix,
                columns=sample_prefix + self.samples.index,
                index=self.guides.index,
            )
            .fillna(0)
            .astype(int)
        )
        if sgrna_column is None:
            mageck_input_df.insert(0, "sgRNA", self.guides.index.tolist())
        elif sgrna_column in self.guides.columns:
            mageck_input_df.insert(0, "sgRNA", self.guides[sgrna_column])
        elif self.guides.index.name == sgrna_column:
            mageck_input_df.insert(0, "sgRNA", self.guides.index.tolist())
        else:
            raise ValueError(f"{sgrna_column} not found in Screen.guides.")
        mageck_input_df["sgRNA"] = mageck_input_df["sgRNA"].map(
            lambda s: s.replace(" ", "_")
        )
        mageck_input_df.insert(1, "gene", self.guides[target_column].astype(str))
        mageck_input_df = mageck_input_df.loc[
            (mageck_input_df.gene.map(lambda o: not pd.isnull(o)))
            & mageck_input_df.gene.map(bool),
            :,
        ]
        if out_path is None:
            return mageck_input_df
        else:
            mageck_input_df.to_csv(out_path, sep="\t", index=False)

    def write(self, out_path):
        """
        Write .h5ad
        """
        super().write(out_path)


def read_h5ad(filename):
    adata = ad.read_h5ad(filename)
    return _Screen.from_adata(adata)


def concat(screens, *args, **kwargs):
    adata = ad.concat(screens, *args, **kwargs)

    return _Screen.from_adata(adata)


def read_csv(X_path=None, guides_path=None, samples_path=None, sep=",", **kwargs):
    if X_path is not None:
        X_df = pd.read_csv(
            X_path,
            delimiter=sep,
            header=0,
            index_col=0,
        )
        X = X_df.values
    else:
        X = None
    guide_df = (
        pd.read_csv(guides_path, sep=sep, **kwargs) if guides_path is not None else None
    )
    samples_df = None if samples_path is None else pd.read_csv(samples_path, sep=sep)
    return _Screen(X=X, guides=guide_df, samples=samples_df)
