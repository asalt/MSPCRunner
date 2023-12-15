import sys
import os
from glob import glob
import re
from pathlib import Path
import pandas as pd
from mokapot import read_pin
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from .commands import RunContainer, Receiver
import ipdb

# from .containers import SampleRun

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

REGX = "(.*[f|F]?\\d)(?=\\.\\d+\\.\\d+\\.\\d+)"


def extract_file_from_scan_header(s: pd.Series):
    return s.str.extract(REGX)


def get_delim(s):
    if "." in s:
        delim = "."
    elif "_" in s:
        delim = "_"
    return delim


def get_scanno(s):
    delim = get_delim(s)
    return s.split(delim)[-2]


def get_filename(s):
    delim = get_delim(s)
    return s.split(delim)[0]


def filechecker(f):
    if f is None:
        return None
    if not os.path.exists(f):
        raise ValueError(f"{f} does not exist")
    return Path(f)


def peek(f, nrows=5):
    df = pd.read_table(f, nrows=nrows)
    if all(x in df for x in MSFRAGGER_TSV_COLUMNS):
        return "msfragger_psms"
    # can add more files for checking here
    return None


class RawResultProcessor:
    """
    THIS IS NOT BEING USED. MERGE WITH below class?
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


def concat(
    search_result_f: str,
    percpsm_f: str,
    sic_f: str,
    ri_f: str,
    scanstats_extra_f: str = None,
    percpept_f: str = None,
):
    """
    script for combining:
    :MSFragger search results:
    :Percolator output:
    :MASIC PrecursorPeaks:
    :MASIC Reporter Ions (if applicable):
    concat
    """
    search_result = pd.read_table(search_result_f).rename(
        columns={
            "ScanNum": "scannum",
            "Peptide": "peptide",
        }
    )

    # percpsm = pd.read_table(percpsm_f, usecols=[0,1,2,3,4,5])

    percpsm = pd.read_table(percpsm_f)

    percpept = None
    if percpept_f is not None:
        percpept = pd.read_table(percpept_f)
    # percpsm = read_pin(str(percpsm_f), to_df=True )
    if sic_f:
        sic = pd.read_table(sic_f)

    PEPT_REGX = re.compile("(?<=\.)(\S+)(?=\.)")

    def get_peptide_sequence(s):
        res = PEPT_REGX.search(s)
        if res is None:
            return s
        peptide = [x for x in res.group() if x.isalpha() and x.isupper()]
        return "".join(peptide)

    if all(x in search_result for x in ("precursor_neutral_mass", "charge")):
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
    if percpept is not None:
        percpept = percpept.rename(
            columns={"peptide": "peptide_modi", "Peptide": "peptide_modi"}
        )

    # TODO clean
    # if 'peptide' in percpsm:
    #    percpsm['peptide'] = percpsm.peptide_modi.apply(get_peptide_sequence)
    # else:
    percpsm["peptide"] = percpsm.peptide_modi.apply(get_peptide_sequence)
    if percpept is not None:
        percpept["peptide"] = percpept.peptide_modi.apply(get_peptide_sequence)
    if "hit_rank" not in search_result:
        search_result["hit_rank"] = 1

    if percpsm_f:
        res = pd.merge(
            search_result,
            percpsm,
            on=["scannum", "peptide"],
            how="right",  # merge right as we don't want decoys
        )
        #
        if percpept is not None:
            _peptide_selection = percpept[(percpept["mokapot q-value"] <= 0.01)].peptide
            logger.info("Controling at 1% peptide FDR")
        else:
            _peptide_selection = percpsm[(percpsm["mokapot q-value"] <= 0.01)].peptide
            logger.info("Controling at 1% psm FDR")

        # res = res[(res["mokapot q-value"] <= 0.01) & (res["hit_rank"] == 1)]
        res = res[(res["peptide"].isin(_peptide_selection)) & (res["hit_rank"] == 1)]
        # res = res[(res["mokapot q-value"] <= 1.01) & (res["hit_rank"] == 1)]
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

    ### ScanStatsEx from MASIC
    ### scan number is the MS3 scan
    ### Master Scan Number is the MS2 scan
    ### master scan number
    # tmp = tmp[tmp["Collision Mode"] == "hcd"]
    # tmp2 = tmp.merge(
    #     sic,
    #     left_on="Master Scan Number",
    #     right_on="SurveyScanNumber",
    #     how="outer",
    #     indicator=True,
    # )
    # tmp2[ tmp2._merge == "both"]

    # scanstats_extra = pd.read_table(scanstats_extra_f)

    if ri_f:
        ri = pd.read_table(ri_f)
        ri = ri.rename(columns={"ScanNumber": "FragScanNumber"})
        # scanstats_extra_f = "./49775_7_ECL_837_TMT_HPLC_F10_MS3-RTS_ScanStatsEx.txt"
        # temporary

        _pat = ri_f.name.strip("ReporterIons.txt") + "*ScanStatsEx.txt"
        # ScanStatsEx may or may not be present. Is necessary for MS3 data
        _fileglobres = list(Path.cwd().glob(_pat))
        # print(_fileglobres)

        assert len(_fileglobres) <= 1
        if _fileglobres:
            scanstats_extra_f = _fileglobres[0]
            #
            scanstats_extra = pd.read_table(scanstats_extra_f)
            scanstats_extra = scanstats_extra.rename(
                columns={"ScanNumber": "FragScanNumber"}
            )
            ri = ri.merge(
                scanstats_extra,
                on=["FragScanNumber", "Collision Mode", "Dataset"],
                # left_on = "ScanNumber",
                # right_on = "ScanNumber",
                how="left",
                # indicator=True,
            )

        res3 = pd.merge(
            res2,
            ri,
            # left_on=["FragScanNumber", "Dataset"],
            # right_on=["Master Scan Number", "Dataset"],
            on=["FragScanNumber", "Dataset"],
            how="left",
        )

    else:
        res3 = res2

    return res3


# MASS_SHIFTS = ['229.1629', '286.184']
MASS_SHIFTS = "229\.1629|286\.184|304.207"


# could consider incorporating in containers.SampleRunContainer
# but it is nice to keep routines separate from container objects
class PSM_Merger(Receiver):
    NAME = "PSM-Merger"
    """
    concatenation of the following:
     - msfragger search results (tsv),
     - masic quant ms1 (tsv)
     - masic quant reporter ions (tsv) if present
     - percolator/mokapot psms (tsv)
    """

    def run(
        self,
        runcontainer: RunContainer = None,
        outdir: Path = None,
        fdr_level="psm",
        force=False,
        **kwargs,
    ) -> List[Path]:  # return ?
        if "force" in kwargs:
            force = kwargs.pop("force")
        if fdr_level is None:
            fdr_level = "psm"
        if fdr_level not in ("psm", "peptide"):
            raise ValueError("must be either psm or peptide")

        if runcontainer is None:
            raise ValueError("No input")
        if outdir is None and runcontainer.rootdir is not None:
            outdir = runcontainer.rootdir
        #
        if outdir is None:
            outdir = Path(".")

        existing_outfile = runcontainer.get_file("MSPCRunner")
        if existing_outfile is not None and existing_outfile.exists() and not force:
            logger.info(f"{existing_outfile} exists for {runcontainer}, skippig")
            return

        outname = os.path.join(
            outdir,
            # runcontainer.rootdir,
            f"{runcontainer.stem}_MSPCRunner_a1.txt",
        )

        search_res = runcontainer.get_file("tsv_searchres")
        # _target_file = "mokapot-psms" if fdr_level == "psm" else "mokapot-peptides" if fdr_level=="peptide" else "mokapot-psms"
        percpept_f = None
        if fdr_level == "peptide":
            percpept_f = runcontainer.get_file("mokapot-peptides")
        percpsm_f = runcontainer.get_file("mokapot-psms")
        sic_f = runcontainer.get_file("SICs")
        ri_f = runcontainer.get_file("ReporterIons")

        logger.info(
            f"""
            search: {search_res}
            percolator: {percpsm_f}
            SIC: {sic_f}
            ReporterIons: {ri_f}
            ---
            outname: {outname}
            """
        )

        if search_res is None:
            logger.info(f"No search result file for {runcontainer.stem}")
            return
        if percpsm_f is None:
            logger.info(f"No percolator result file for {runcontainer.stem}")
            return

        df = concat(
            search_result_f=search_res,
            sic_f=sic_f,
            percpsm_f=percpsm_f,
            ri_f=ri_f,
            percpept_f=percpept_f or None,
        )
        df = df.rename(
            columns={
                "scannum": "First Scan",
                "parent_charge": "Charge",
                "mokapot q-value": "q_value",
                "hyperscore": "Hyperscore",
                "deltaRank1Rank2Score": "DeltaScore",
                "Calibrated Observed M/Z": "mzDa",
                "tot_num_ions": "TotalIons",
                "hit_rank": "Rank",
            }
        )
        # fix specid

        regx = r"(.*)(?=\.\d+\.\d+\.\d+_\d+)"
        df["SpectrumFile"] = extract_file_from_scan_header(df["SpecId"])

        # outname = f"{basename}_percolator_MASIC.txt"

        maybe_calc_labeling_efficiency(df, outname)
        maybe_calc_phos_enrichment(df, outname)

        print(f"Writing {outname}")
        df.to_csv(outname, sep="\t", index=False)
        # here we shoudl create and return a SampleRun object?
        # sr = SampleRun()
        return


def maybe_calc_labeling_efficiency(df, outname):
    if "modification_info" not in df:
        return
    if not df.modification_info.fillna("").str.contains(MASS_SHIFTS).any():
        return
    with open("labeling_efficiency.txt", "a") as f:
        labeled = df[df.modification_info.fillna("").str.contains(MASS_SHIFTS)]
        f.write(f"{outname}\t{len(labeled)}\t{len(df)}\t{len(labeled)/len(df)}\n")


def maybe_calc_phos_enrichment(df, outname):
    if "modification_info" not in df:
        return
    if not df.modification_info.fillna("").str.contains("79\.96").any():
        return
    with open("phospho_enrichment.txt", "a") as f:
        phos = df[df.modification_info.fillna("").str.contains("79\.96")]
        f.write(f"{outname}\t{len(phos )}\t{len(df)}\t{len(phos )/len(df)}\n")


def main(path=None, fdr_level=None):
    if fdr_level is None:
        fdr_level = "psm"
    if fdr_level not in ("psm", "peptide"):
        raise ValueError("must be either psm or peptide")
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
