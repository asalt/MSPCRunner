import sys
import os
from glob import glob
import re
from pathlib import Path
import pandas as pd
from mokapot import read_pin
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from .commands import RunContainer

from .logger import get_logger

logger = get_logger(__name__)

MSFRAGGER_TSV_COLUMNS = (
    "scannum",
    "precursor_neutral_mass",
    "retention_time",
    "charge",
    "hit_rank",
    "peptide",
    "peptide_prev_aa",
    "peptide_next_aa",
    "protein",
    "num_matched_ions",
    "tot_num_ions",
    "calc_neutral_pep_mass",
    "massdiff",
    "num_tol_term",
    "num_missed_cleavages",
    "modification_info",
    "hyperscore",
    "nextscore",
    "expectscore",
    "best_locs",
    "score_without_delta_mass",
    "best_score_with_delta_mass",
    "second_best_score_with_delta_mass",
    "delta_score",
    "alternative_proteins",
)


def get_scanno(s):
    return s.split(".")[-2]


def get_filename(s):
    return s.split(".")[0]


def filechecker(f):
    if f is None:
        return None
    if not os.path.exists(f):
        raise ValueError(f"{f} does not exist")
    return Path(f)


def peak(f, nrows=5):
    df = pd.read_table(f, nrows=nrows)
    if all(x in df for x in MSFRAGGER_TSV_COLUMNS):
        return "msfragger_psms"
    # can add more files for checking here
    return None


class RawResultProcessor:
    """
    container for combining:
    :MSFragger (or other?) search results:
    :Percolator output:
    :MASIC PrecursorPeaks:
    :MASIC Reporter Ions (if applicable):
    concat
    """

    def __init__(
        self,
        psm_file=None,
        percolator_file=None,
        sic_file=None,
        ri_file=None,
        outpath=None,
    ):

        # for f in (psm_file, percolator_file, sic_file, ri_file):

        self.psm_file = filechecker(psm_file)
        self.percolator_file = filechecker(percolator_file)
        self.sic_file = filechecker(sic_file)
        self.ri_file = filechecker(ri_file)

        self._name = None
        self.result = None

        if self.psm_file is not None and outpath is None:
            self.outpath = self.psm_file.parent
        else:
            self.outpath = Path(".")

    @property
    def name(self):
        if self._name is None:
            if self.psm_file is None:
                self._name = "<None>"
            else:
                self._name = os.path.splitext(self.psm_file.name)[0]
        return self._name

    def __repr__(self):
        return f"RawResultProcessor {self.name}"

    def concat(self):

        res = concat(self.psm_file, self.percolator_file, self.sic_file, self.ri_file)
        self.result = res
        return

    def save(self):

        outname = self.outpath / f"{self.name}_percolator_MASIC.txt"
        print(f"Writing {outname}")
        self.result.to_csv(outname, sep="\t", index=False)
        return


def concat(search_result_f, percpsm_f, sic_f, ri_f):

    """
    script for combining:
    :MSFragger search results:
    :Percolator output:
    :MASIC PrecursorPeaks:
    :MASIC Reporter Ions (if applicable):
    concat
    """

    search_result = pd.read_table(search_result_f)
    # percpsm = pd.read_table(percpsm_f, usecols=[0,1,2,3,4,5])
    percpsm = pd.read_table(percpsm_f)
    # percpsm = read_pin(str(percpsm_f), to_df=True )
    if sic_f:
        sic = pd.read_table(sic_f)
    if ri_f:
        ri = pd.read_table(ri_f)

    PEPT_REGX = re.compile("(?<=\.)(\S+)(?=\.)")

    def get_peptide_sequence(s):
        res = PEPT_REGX.search(s)
        if res is None:
            return s
        peptide = [x for x in res.group() if x.isalpha() and x.isupper()]
        return "".join(peptide)

    search_result["mz"] = (
        search_result.precursor_neutral_mass + search_result.charge
    ) / search_result.charge
    # percpsm['filename'] = percpsm.PSMId.apply(get_filename)
    # percpsm['scannum'] = percpsm.PSMId.apply(get_scanno).astype(int)
    percpsm["filename"] = percpsm.SpecId.apply(get_filename)
    percpsm["scannum"] = percpsm.SpecId.apply(get_scanno).astype(int)
    percpsm = percpsm.rename(
        columns={"peptide": "peptide_modi", "Peptide": "peptide_modi"}
    )

    # TODO clean
    # if 'peptide' in percpsm:
    #    percpsm['peptide'] = percpsm.peptide_modi.apply(get_peptide_sequence)
    # else:
    percpsm["peptide"] = percpsm.peptide_modi.apply(get_peptide_sequence)

    if percpsm_f:
        res = pd.merge(
            search_result,
            percpsm,
            on=["scannum", "peptide"],
            how="right",  # merge right as we don't want decoys
        )

        res = res[(res["mokapot q-value"] <= 0.01) & (res["hit_rank"] == 1)]
    else:
        res = search_result

    if sic_f:
        res2 = pd.merge(
            res.query("hit_rank==1"),
            sic,
            left_on=["scannum"],
            right_on=["FragScanNumber"],
            how="left",
        )
    else:
        res2 = res

    if ri_f:
        res3 = pd.merge(
            res2,
            ri,
            left_on=["scannum", "Dataset"],
            right_on=["ScanNumber", "Dataset"],
            how="left",
        )
    else:
        res3 = res2

    return res3


# MASS_SHIFTS = ['229.1629', '286.184']
MASS_SHIFTS = "229\.1629|286\.184"


class PSM_Merger:

    NAME = "PSM-Merger"
    """
    concatenation of the following:
     - msfragger search results (tsv),
     - masic quant ms1 (tsv)
     - masic quant reporter ions (tsv) if present
     - percolator/mokapot psms (tsv)
    """

    def run(
        self, runcontainer: RunContainer = None, outdir: Path = None, **kwargs
    ) -> List[Path]:

        if runcontainer is None:
            raise ValueError("No input")

        search_res = runcontainer.get_file("tsv_searchres")
        percpsm_f = runcontainer.get_file("mokapot-psms")
        sic_f = runcontainer.get_file("SICs")
        ri_f = runcontainer.get_file("ReporterIons")

        if search_res is None:
            logger.info(f"No search result file for {runcontainer.stem}")
            return
        if percpsm_f is None:
            logger.info(f"No search result file for {runcontainer.stem}")
            return

        df = concat(
            search_result_f=search_res, sic_f=sic_f, percpsm_f=percpsm_f, ri_f=ri_f
        )
        # outname = f"{basename}_percolator_MASIC.txt"
        outname = f"{runcontainer.stem}_MSPCRunner_a1.txt"

        maybe_calc_labeling_efficiency(df, outname)

        print(f"Writing {outname}")
        df.to_csv(outname, sep="\t", index=False)

        1 + 1
        1 + 3


def maybe_calc_labeling_efficiency(df, outname):
    if not df.modification_info.fillna("").str.contains(MASS_SHIFTS).any():
        return
    with open("labeling_efficiency.txt", "a") as f:
        labeled = df[df.modification_info.fillna("").str.contains(MASS_SHIFTS)]
        f.write(f"{outname}\t{len(labeled)}\t{len(df)}\t{len(labeled)/len(df)}\n")


def main(path=None):
    if len(sys.argv) < 2 and path is None:
        print("USAGE python psm_merge.py <target_directory>")
        sys.exit(0)
    if path is None:
        path = sys.argv[1]

    path = Path(path)

    # files = glob(os.path.join(path, '*tsv'))
    files = path.glob("*tsv")
    for f in files:
        # basename = os.path.splitext(f)[0]
        # print(files)

        # sic_f = glob(f"{basename}_SICstats.txt")
        sic_f = list(path.glob(f"{f.stem}*SICstats.txt"))
        if sic_f:
            sic_f = sic_f[0]

        # percpsm_f = glob(f'{basename}.pin-percolator-psms.txt')
        # percpsm_f = glob(f'{basename}*mokapot.psms.txt')
        percpsm_f = list([x for x in path.glob(f"{f.stem}*mokapot.psms.txt")])
        # print(x for x in percpsm_f)
        if not percpsm_f:
            print(f"Could not find percolator psm file for {f}")
            continue
        percpsm_f = percpsm_f[0]

        ri_f = list(path.glob(f"{f.stem}_ReporterIons.txt"))
        if ri_f:
            ri_f = ri_f[0]
        else:
            ri_f = None

        df = concat(search_result_f=f, sic_f=sic_f, percpsm_f=percpsm_f, ri_f=ri_f)
        # outname = f"{basename}_percolator_MASIC.txt"
        outname = f"{f.stem}_MSPCRunner_a1.txt"

        maybe_calc_labeling_efficiency(df, outname)

        print(f"Writing {outname}")
        df.to_csv(outname, sep="\t", index=False)


if __name__ == "__main__":

    main()
