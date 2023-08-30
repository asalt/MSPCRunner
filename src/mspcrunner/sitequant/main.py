# main.py
"""
need to fix cases where total mass shift is a combination of two mods
need to ensure this works for label free
"""
import os
import re
from pathlib import Path
import warnings
import logging
from collections import defaultdict
from copy import deepcopy as copy
from functools import partial
from typing import Iterable, Tuple
from RefProtDB.utils import fasta_dict_from_file


import pandas as pd
import click
import enlighten

from runner import runner


# modi_dict = {
#     79.96633: "p",
#     15.994915: "o",
#     #  "114.04293": 'gg'
#     114.042927: "gg", # ubiquitin label on lysine (gg remnant)
#     114.04293: "gg", # ubiquitin label on lysine (gg remnant)
#     15.994915: "o",  # Oxidation on methionine
#     42.010565: "a",  # Acetylation on N-terminus
#     304.207: "t",    # TMTpro label on N-terminus or lysine
#     -17.02650: "q",  # Pyro-Glu from glutamine or cyclization of N-terminal glutamine
#     -18.01060: "e",  # Pyro-Glu from glutamic acid
#     320.2019: "at",  # Likely a combination of acetylation and TMT label
#     418.24994: "gt", # Likely a combination of gg remnant and TMT label
#     361.22845: "ot", # Likely a combination of oxidation and TMT label
#     608.414: "tt",   # TMTpro label on both N-terminus and lysine
#     722.4569: "??",  # Unknown, possibly a combination of multiple modifications
#     286.1964: "??",   # Unknown, possibly a combination of multiple modifications
# }

DEFAULT_MODI_DICT = {
    79.96633: {'name': 'p', 'possible_aa': ('S', 'T', 'Y')},
    15.994915: {'name': 'o', 'possible_aa': ('M',)},
    114.042927: {'name': 'gg', 'possible_aa': ('K',)},
    114.04293: {'name': 'gg', 'possible_aa': ('K',)},
    42.010565: {'name': 'a', 'possible_aa': ('Any (N-term)',)},
    304.207: {'name': 't', 'possible_aa': ('K', 'Any (N-term)')},
    -17.02650: {'name': 'q', 'possible_aa': ('Q', 'Any (N-term)')},
    -18.01060: {'name': 'e', 'possible_aa': ('E',)},
    320.2019: {'name': 'at', 'possible_aa': ('M', 'Any (N-term)')},  # Needs further investigation
    418.24994: {'name': 'gt', 'possible_aa': ('K',)},  # Needs further investigation
    361.22845: {'name': 'ot', 'possible_aa': ('M', 'K')},  # Needs further investigation
    608.414: {'name': 'tt', 'possible_aa': ('K', 'Any (N-term)')},  # Needs further investigation
    722.4569: {'name': '??', 'possible_aa': ()},  # Unknown
    286.1964: {'name': '??', 'possible_aa': ()},  # Unknown
}


class ModiTypeAssigner:
    def __init__(self, modi_dict=DEFAULT_MODI_DICT):
        # Multi-level dictionary for ModiType mapping
        self.modi_dict = modi_dict

    def get_modi_type(self, mass_shift, aa):
        if isinstance(mass_shift, str):
            mass_shift = float(mass_shift)
        mass_shift = round(mass_shift, 6)  # Rounding to avoid floating-point errors
        modi_info = self.modi_dict.get(mass_shift, {'name': '?', 'possible_aa': []})
        if aa in modi_info['possible_aa']:
            #modi_name 
            modi_name = modi_info['name'] 
        else:
            modi_name = '?'
        return modi_name 

def make_nr(df) -> list([pd.DataFrame]):
    logging.info("makenr")
    id_cols = [
        "Site",
        "quality",
        "AA",
        "Modi",
        "GeneID",
        "basename",
    ]  #'label']

    def condense_group(group, idxs, df, id_cols=id_cols):
        # idx = [0, 1]
        sel = df.loc[idxs]
        sel["Protein"] = sel["Protein"].fillna("")
        proteins = str.join(", ", map(str, sel["Protein"].tolist()))
        aapos_list = str.join(", ", map(str, set(sel["AApos"].tolist())))

        try:
            primarysel = (
                str.join(
                    ",",
                    (
                        sel.apply(
                            lambda x: x["Protein"]
                            if x["Primary_select"] == "Yes"
                            else "",
                            axis=1,
                        )
                    ),
                ).strip(",")
                or None
            )
        except Exception as e:
            pass

        secondarysel = (
            str.join(
                ",",
                (
                    sel.apply(
                        lambda x: x["Protein"]
                        if x["Secondary_select"] == "Yes"
                        else "",
                        axis=1,
                    )
                ),
            ).strip(",")
            or None
        )
        #
        uniquesel = None
        bestsel = None
        if (
            bool(primarysel) == False
            and bool(secondarysel) == False
            and proteins.count(",") == 0
        ):
            uniquesel = proteins
            bestsel = None
        elif (
            bool(primarysel) == False
            and bool(secondarysel) == False
            and proteins.count(",") > 0
        ):
            bestsel = sel.sort_values(by="protein_length", ascending=False).iloc[0][
                "Protein"
            ]
            uniquesel = None

        finalsel = None
        reason = None
        if bool(primarysel):
            finalsel = primarysel
            reason = "primary"
        elif not bool(primarysel) and bool(secondarysel):
            finalsel = secondarysel
            reason = "secondary"
        elif uniquesel is not None:
            finalsel = uniquesel
            reason = "not primary or secondary, only one choice"
        elif bestsel is not None:
            finalsel = bestsel
            reason = "not primary or secondary, multiple choices, returning longest"

        # import ipdb; ipdb.set_trace()

        #

        # str.join(", ", map(str, set(sel["Primary_select"].tolist())))
        # make unique before making string
        d = dict()
        d["Protein_Accessions"] = proteins
        d["AApos_List"] = aapos_list
        d["Primary_select"] = primarysel or ""
        d["Secondary_select"] = secondarysel or ""
        d["Unique_select"] = uniquesel or ""
        d["Best_select"] = bestsel or ""
        d["Protein_Accession_Selected"] = finalsel
        d["Reason_Selected"] = reason
        for k, v in zip(id_cols, group):
            d[k] = v

        # add quant values
        _cols = set(df.columns) - set(d.keys())
        _cols = [
            x
            for x in _cols
            if x not in ("AApos", "Protein", "Protein_id", "protein_length")
        ]
        # avoid some columns from the original table we do not want to add
        # in fact the only columns we want to add are the ones that start with TMT_ (or something else if it isn't a TMT experiment)

        for col in _cols:
            d[col] = sel.iloc[0][col]

        return d

    def n_before_c(c: str):
        """
        sort function
        'TMT_127_N' -> 1270
        'TMT_126' -> 1260
        'TMT126' -> 1260
        '134' -> 1340
        """
        c = str(c)
        c = c.strip("TMT").strip("_")
        # a = re.match(r"(\d+)", c).groups()
        numbers = re.findall(r"(\d+)", c)
        # numbers = list(map(int, numbers))
        if c[-1].isalpha():
            val = {"N": "0", "C": "1"}.get(c[-1])
            numbers.append(val)
        unique_id = str.join("", map(str, numbers))
        res = int(unique_id)
        return res

    # n_before_c("127_C")
    # n_before_c("127_N")
    # n_before_c("134")
    n_before_c("51942_1_7_126_N")

    # could do a check to ensure unmiqueness
    # assert (df.groupby(id_cols).label.value_counts() == 1).all()
    # df = pd.pivot(df, index=id_cols, columns="label", values="quantity").reset_index()
    # other_cols = ['AApos', 'protein', ]
    # id_cols = ["Site", "quality", "AA", "Modi", "GeneID", "basename", ]#'label']
    g = df.groupby(id_cols)
    groups = g.groups
    res = list()
    for group, idxs in groups.items():
        d = condense_group(group, idxs, df, id_cols=id_cols)
        res.append(d)

    out = pd.DataFrame(res).rename(
        columns={
            "Protein": "Protein_Accessions",
            "Modi": "MassShift",
            "quality": "TopScore",
            "Symbol": "GeneSymbol",
            "Description": "GeneDescription",
            # "Final_select" : "Protein_Accession_Selected",
            # "Reason" : "Reason_Selected",
            # "AApos" : "AApos_Selected",
        }
    )
    rec_run_search = re.compile(r"^(?P<recno>\d{5,})_(?P<runno>\d+)_(?P<searchno>\d+)")
    _match = out["basename"].str.extractall(rec_run_search).reset_index()
    assert (_match.match == 0).all()
    assert (_match.index == _match.level_0).all()
    _match = _match.drop(columns=["match", "level_0"])
    _rec_run_search_value = (
        f"{_match.iloc[0].recno}_{_match.iloc[0].runno}_{_match.iloc[0].searchno}"
    )
    out = pd.concat([out, _match], axis=1)

    # rename each of the TMT_1xx quant columns as as rec_run_search_label
    _renamer = dict()
    for x in out.columns:
        if x.startswith("TMT_") or x == "none":
            _renamer[x] = f"{_rec_run_search_value}_{x.strip('TMT_')}"
            # out = out.rename(columns={x: f"{x}_{out.iloc[0]['basename']}"})
    out = out.rename(columns=_renamer)

    # it may be better to use a dictionary to map dfall[['AApos']] to df_nr
    # or do this:
    _mapping = df.drop_duplicates(["Protein", "protein_length", "AApos", "Site"])[
        ["Protein", "protein_length", "AApos", "Site"]
    ]
    _out = pd.merge(
        out,
        _mapping,
        left_on=["Protein_Accession_Selected", "Site"],
        right_on=["Protein", "Site"],
        how="left",
    )
    _out["AApos"] = _out["AApos"].fillna(0).astype(int)
    _out["protein_length"] = _out["protein_length"].fillna(0).astype(int)
    _out = _out.rename(
        columns={"AApos": "AApos_Selected", "protein_length": "Protein_Length_Selected"}
    )
    # import ipdb ; ipdb.set_trace()
    _out = _out.drop(columns=["Protein"])
    out = _out

    #
    # import ipdb; ipdb.set_trace()
    # Instantiate the class and use method to assign ModiType
    modi_assigner = ModiTypeAssigner()
    out["ModiType"] = out.apply(lambda x: modi_assigner.get_modi_type(x["MassShift"], x["AA"]), axis=1)


    # out["ModiType"] = out.apply(lambda x: assign_modi_type(x, modi_dict), axis=1)
    #out["ModiType"] = out.apply(lambda x: modi_dict.get(x["MassShift"], "?"), axis=1)
    out["SiteName"] = out.apply(
        lambda x: f"{x['GeneSymbol']}_{x['ModiType']}{x['AA']}{x['AApos_Selected']}",
        axis=1,
    )
    out["SiteID"] = out.apply(
        lambda x: f"{x['GeneID']}_{x['Protein_Accession_Selected']}_{x['ModiType']}{x['AA']}{x['AApos_Selected']}_{x['Site']}",
        axis=1,
    )

    # "_".join([x["GeneSymbol"], x["ModiType"], x["AA"], x["AApos"]]), axis=1)

    #
    # qual_cols = set(df.columns) - {*set(id_cols), "AApos", "Protein", "Description"}
    # qual_cols = tuple(qual_cols)
    # qual_cols = sorted(qual_cols, key=n_before_c)
    col_order = [
        "recno",  # delete from final
        "runno",  # delete from final
        "searchno",  # delete from final
        "basename",  # delete from final
        "SiteName",  # (GeneSymbol)_(type of modi)(AA)(AApos_Selected)
        "SiteID",  # (GeneID)_(Protein_Accesion_Selected)_(type of modi)
        "GeneID",
        "GeneSymbol",
        "GeneDescription",
        "Site",
        "AA",
        "MassShift",
        "ModiType",  # delete from final
        "Primary_select",  # delete from final
        "Secondary_select",  # delete from final
        "Unique_select",  # delete from final
        "Best_select",  # delete from final
        "Protein_Accession_Selected",
        "Reason_Selected",
        "Protein_Length_Selected",
        # "basename",
        "AApos_Selected",
        "Protein_Accessions",
        "AApos_List",
        "TopScore",
        # *quant_cols
        # *sorted(set(df.columns) - set(id_cols)),
    ]
    quant_cols = set(out.columns) - set(col_order)
    quant_cols = sorted(quant_cols, key=n_before_c)
    col_order = col_order + quant_cols
    out = out[col_order]

    return out


from rpy2.robjects.packages import importr
from rpy2.robjects import r


def write_gct(df, outname):
    """
    df is nr output
    """

    from rpy2.robjects.packages import importr
    from rpy2.robjects import r
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()

    cmapR = importr("cmapR")
    # data_obj = ctx.obj["data_obj"]
    quant_cols = [
        x for x in df.columns if re.match("^\d+_\d+_\d+.*$", x) or x == "none"
    ]

    m = df[quant_cols]
    # import ipdb; ipdb.set_trace()
    rid = df.SiteID
    rdesc_cols = set(df.columns) - set(quant_cols)
    assert "SiteID" in rdesc_cols
    rdesc = df[rdesc_cols].set_index("SiteID")
    cid = quant_cols
    # cdesc

    r.assign("m", m)
    r.assign("rid", rid)
    r.assign("rdesc", rdesc.fillna(""))
    # r.assign("cdesc", cdesc)
    r.assign("cid", pd.Series(cid))
    r.assign("outname", outname)
    # my_new_ds <- new("GCT", mat=m)
    # r('my_ds <- new("GCT", mat=as.matrix(m), rid=rid, cid=cid, cdesc=cdesc, rdesc=as.data.frame(rdesc))')
    r(
        'my_ds <- new("GCT", mat=as.matrix(m), rid=rid, cid=cid, rdesc=as.data.frame(rdesc))'
    )
    r(
        'write_gct(my_ds, file.path(".", outname), precision=8)'
    )  # r doesn't keep the path relative for some reason

    return

    # note here is where df_nr_mednorm was being produced previously

    # qual_cols = set(df.columns) - {*set(id_cols), "AApos", "Protein", "Description"}
    # qual_cols = tuple(qual_cols)
    # qual_cols = sorted(qual_cols, key
    # =n_before_c)
    # # q = out.iloc[0]["basename"]
    # # rec_run_search.findall()
    # # rec_run_search

    # col_order = [
    #     "recno",
    #     "runno",
    #     "searchno",
    #     "GeneID",
    #     "Description",
    #     "Site",
    #     "AA",
    #     "MassShift",
    #     "TopScore",
    #     # "basename",
    #     "AApos",
    #     "Protein_Accessions",
    #     *qual_cols
    #     # *sorted(set(df.columns) - set(id_cols)),
    # ]
    # out = out[col_order]
    # out2 = out.copy()
    # for col in qual_cols:
    #     out2[col] = out[col] / out[col].median()

    # return out, out2


def parse_rawname(name: str) -> Tuple[str, str, str]:
    """yield up to the first 3 numbers in a string separated by underscore
    returns None when number is missing / interrupted
    """
    namesplit = str(name).split("_")
    yield_counter = 0
    for x in namesplit:
        if x.isnumeric() and yield_counter < 3:
            yield_counter += 1
            yield x
        else:
            break
    while yield_counter < 3:
        yield_counter += 1
        yield None


def maybe_split_on_proteinid(df):
    # look at geneid column
    SEP = ";"
    if "Proteins" not in df:
        return df
    if not df["Proteins"].str.contains(SEP).any() and "GeneID" in df:
        return df

    glstsplitter = (
        df["Proteins"]
        .str.split(SEP)
        .apply(pd.Series, 1)
        .stack()
        .to_frame(name="GeneID")
    )

    glstsplitter.index = glstsplitter.index.droplevel(-1)  # get rid of
    # multi-index
    df = df.join(glstsplitter).reset_index()
    df["GeneID"] = df.GeneID.fillna("-1")
    df.loc[df.GeneID == "", "GeneID"] = "-1"
    df["GeneID"] = df.GeneID.fillna("-1")
    # df['GeneID'] = df.GeneID.astype(int)
    df["GeneID"] = df.GeneID.astype(str)

    if "index" in df:
        df = df.drop("index", axis=1)
    return df


def remove_contaminants(df):
    return df[
        ~(df.GeneID.str.startswith("CON"))
        | ~(df.GeneID.str.startswith("cont_"))
        | ~(df.GeneID.str.startswith("con"))
    ]


def wide_to_long_xinpei_table(df):
    id_vars = [
        "Title",
        "Charge",
        "Sequence",
        "Proteins",
        "Modifications",
        "SequenceModi",
        "PeakArea",
        "ParentIonIntensity",
        "GeneID",
    ]

    res = df.melt(
        id_vars=id_vars,
        value_name="SequenceArea",
    ).rename(columns={"variable": "LabelFLAG"})
    return res


def preprocess(df):
    df = df.rename(
        columns={
            k: k.split("psm_")[1]
            for k in [col for col in df.columns if col.startswith("psm_")]
        },
        # inplace=True,
    )
    if "Modifications" not in df.columns:
        df = df.rename(columns={"Modifications_abbrev": "Modifications"})
    df = df.rename(columns={"Peptide": "Sequence"})
    # import ipdb; ipdb.set_trace()
    df = maybe_split_on_proteinid(df)
    df["GeneID"] = df.GeneID.astype(str)
    df = remove_contaminants(df)
    # special case
    # df = wide_to_long_xinpei_table(df)

    # =======
    if "PSM_UseFLAG" not in df:
        df["PSM_UseFLAG"] = 1
    if "IonScore" not in df:
        df["IonScore"] = -1

    df["GeneID"] = df["GeneID"].astype(str)
    df["Modifications"] = df["Modifications"].astype(str).fillna("")

    return df


def load(p: Path, data_dir: Path):
    if p.exists() and p.is_file():
        df = pd.read_table(p, dtype={"GeneID": "str", "psm_GeneID": "str"})
        logging.info(f"Loaded {p}: {df.shape}")

    else:  # try ispec.
        logging.info(f"Loading {p} with bcmproteomics.ispec.PSMs")
        from bcmproteomics_ext import ispec

        rec, run, search = parse_rawname(p)
        exp = ispec.PSMs(rec, run, search, data_dir=data_dir)
        df = exp.df
        # df = df.sample(100)
        # df = exp.df.head(40)
        # df = check_for_dup_cols(df)
    # df = df.sample(200)
    return df


@click.command()
@click.option("--cores", default=1, show_default=True)
@click.option("-p", "--psms", type=Path, multiple=True)
@click.option("-o", "--out", default=None)
@click.option("--data-dir", default=".")
@click.option(
    "-f", "--fasta", type=click.Path(exists=True, dir_okay=False), help="fasta file"
)
def main(cores, psms, out, data_dir, fasta, **kwargs):
    logging.basicConfig(level=logging.INFO)
    if fasta is None:
        raise ValueError("must supply fasta")

    fa = pd.DataFrame.from_dict(fasta_dict_from_file(fasta, "specific"))
    # ensure "geneid" column is filled
    _ix = fa.loc[fa.geneid.isna()].index
    fa.loc[_ix, "geneid"] = fa.raw_header

    # if "ENSP" in fa.columns:
    # _ix = fa.loc[fa.ref.isna()].index
    # _to_replace = fa_generic.loc[_ix].header
    # fa.loc[_ix, "ref"] = _to_replace
    fasta_df = fa
    # import ipdb; ipdb.set_trace()

    for p in psms:
        logging.info(f"working on {p}")

        df = load(p, data_dir=data_dir)
        df = preprocess(df)
        # df = df
        geneids = df.GeneID.unique()
        # geneids = list(geneids)[0:100]
        df = df[df.GeneID.isin(geneids)]
        # geneids = ['gi|290766066|gb|ADD60044.1|',]
        # geneids = geneids[:1]
        logging.info(f"{len(geneids)} genes")

        if out is None:
            basename = os.path.split(os.path.splitext(p)[0])[1]
            # basename = '{}_{}_{}'.format(os.path.split(os.path.splitext(p)[0])[1], g, label)
        else:
            basename = f"{out}"

        if cores > 1:
            from multiprocessing import Pool, set_start_method

            try:
                set_start_method("spawn")
            except RuntimeError:
                pass
            runner_partial = partial(
                runner,
                df=df,
                fasta=fasta_df,
                basename=basename,
            )

            ALL_RESULTS = list()
            # with Pool(processes=cores) as pool, logging_redirect_tqdm():

            with Pool(processes=cores) as pool:
                # ALL_RESULTS = pool.map(run, arglist)
                # for res in tqdm.tqdm(

                pbar = enlighten.Counter(
                    total=df.GeneID.nunique(), desc="pbar", unit="tick"
                )
                for res in pool.imap_unordered(runner_partial, geneids):  # cores
                    # for res in pool.imap_unordered(runner, geneids=geneid, df=df, fa=fa, basename=basename, make_plot=plot, combine=combine, cores=cores, chunksize=1e2):
                    ALL_RESULTS.append(res)
                    pbar.update()
                # ALL_RESULTS = pool.imap_unordered(runner_partial, geneid[:510], cores)
                ALL_RESULTS = [
                    x for y in ALL_RESULTS for x in y
                ]  # unpack list of lists
        else:  # cores == 1
            # try to get all results
            try:
                ALL_RESULTS = runner(
                    geneids=geneids,
                    df=df,
                    fasta=fasta_df,
                    basename=basename,
                )
            except Exception as e:
                logging.error(e)
                ALL_RESULTS = list()

        dfall = pd.concat(ALL_RESULTS)
        id_cols = [x for x in dfall.columns if x not in ("quantity", "label")]
        assert (dfall.groupby(id_cols).label.value_counts() == 1).all()
        # id_cols = [x for x in dfall.columns if x not in ("quantity", "label", "quality")]
        _dfall = dfall.pivot(
            index=id_cols, columns="label", values="quantity"
        ).reset_index()
        dfall = _dfall

        # now match to primary / secondary isoform and position
        # two new columns
        # column 1 is site type: primary, secondary, non selected unique, and non selected best
        # column 2 is site name: example EGFR_pY922
        # column 3 is the main, chosen, protein accession
        # column 4 is the main selected position
        #
        def add_prioritize_site_table(dfall, ref):
            res = pd.merge(
                dfall,
                ref[["Primary_select", "Secondary_select", "Protein_id"]],
                left_on="Protein",
                right_on="Protein_id",
                how="left",
            )
            return res

        _path = os.path.dirname(os.path.realpath(__file__))
        _f = os.path.join(_path, "GENCODE.V42.basic.CHR.isoform.selection.mapping.txt")
        _df1 = pd.read_csv(_f, sep="\t")
        _f = os.path.join(_path, "GENCODE.M32.basic.CHR.protein.selection.mapping.txt")
        _df2 = pd.read_csv(_f, sep="\t")
        _df = pd.concat([_df1, _df2])
        dfall_with_priority_site = add_prioritize_site_table(dfall, _df)
        dfall = dfall_with_priority_site

        outname = os.path.join(data_dir, f"{basename}_site_table.tsv")

        # pivot here
        logging.info(f"Writing {outname}")
        dfall.to_csv(outname, sep="\t", index=False, mode="w")
        #

        #

        logging.info("making non redundant table")
        # import ipdb; ipdb.set_trace()
        df_nr = make_nr(dfall)

        # _f = 79.96633
        # logging.info(f"Filtering mass shift for {_f}")
        # df_nr = df_nr[df_nr.MassShift == _f]

        outname = os.path.join(data_dir, f"{basename}_site_table_nr.tsv")
        df_nr.to_csv(outname, sep="\t", index=False, mode="w")
        write_gct(df_nr, outname=outname)
        logging.info(f"wrote {outname}")

        # _idx = ["recno", "runno", "searchno", "label"]
        # median_values = df_nr.groupby(_idx).quantity.transform("median")
        # df_nr_mednorm = df_nr.copy()
        # df_nr_mednorm['quantity'] = df_nr_mednorm['quantity'] / median_values
        # # med norm
        # outname = os.path.join(data_dir, f"{basename}_site_table_nr_mednorm.tsv")
        # df_nr_mednorm.to_csv(outname, sep="\t", index=False, mode="w")
        # logging.info(f"wrote {outname}")

        outname = os.path.join(
            data_dir,
            f"{basename}_site_table_GENCODE.V42.basic.CHR.isoform.selection.mapping.txt.tsv",
        )
        dfall_with_priority_site.to_csv(outname, sep="\t", index=False)
        # now match to values

        # values_col = "quantity"
        # name_col = "label"

        # import ipdb; ipdb.set_trace()
        # now pivot for label
        # _theindex = [x for x in dfall if x not in (values_col, name_col)]
        # if len(set(_theindex)) != len(_theindex):
        #     logging.warning(f"Problem with the index")

        # let's not pivot here - yet
        # dfall = dfall.pivot(
        #     #index=[x for x in dfall if x not in (values_col, name_col)],
        #     index=_theindex,
        #     values=values_col,
        #     columns=name_col,
        # ).reset_index()

        # need to make df_nr AND df_nr_mednorm here....
        #
        #


if __name__ == "__main__":
    main()
