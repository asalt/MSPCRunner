# main.py
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

def make_nr(df) -> list([pd.DataFrame]):

    logging.info('makenr')
    id_cols = ["Site", "quality", "AA", "Modi", "GeneID", "basename", ]#'label']

    def condense_group(group, idxs, df, id_cols=id_cols):
        # idx = [0, 1]
        sel = df.loc[idxs]
        proteins = str.join(", ", map(str, sel["Protein"].tolist()))
        aapos_list = str.join(", ", map(str, set(sel["AApos"].tolist())))

        primarysel = str.join(',', (sel.apply(lambda x : x['Protein'] if x['Primary_select']=='Yes' else '', axis=1))).strip(',') or None
        secondarysel = str.join(',', (sel.apply(lambda x : x['Protein'] if x['Secondary_select']=='Yes' else '', axis=1))).strip(',') or None
        #
        uniquesel = None
        bestsel = None
        if bool(primarysel) == False and bool(secondarysel) == False and proteins.count(',') == 0:
            uniquesel = proteins
            bestsel = None
        else:
            bestsel = sel.sort_values(by="protein_length", ascending=False).iloc[0]["Protein"]
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



        #

        #str.join(", ", map(str, set(sel["Primary_select"].tolist())))
        # make unique before making string
        d = dict()
        d["Protein_Accessions"] = proteins
        d["AApos_list"] = aapos_list
        d["Primary_select"] = primarysel or ""
        d["Secondary_select"] = secondarysel or ""
        d["Unique_select"] = uniquesel or ""
        d["Best_select"] = bestsel or ""
        d["Final_select"] = finalsel
        d["Reason"] = reason
        for k, v in zip(id_cols, group):
            d[k] = v

        # add quant values
        _cols = set(df.columns) - set(d.keys())
        _cols = [x for x in _cols if x not in ('AApos', 'Protein', 'Protein_id', 'protein_length')]
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
        a = re.match(r"(\d+)", c).groups()
        if a is None:
            # import ipdb; ipdb.set_trace()
            1+1
        else:
            a = a[0]
        b = "0"
        if c[-1].isalpha():
            b = {"N": "0", "C": "1"}.get(c[-1])
        cats = re.match(r"_?(\d+)[_(\w)]?", c).groups()
        a = cats[0]
        getint = lambda x: {"N": "0", "C": "1"}.get(x)
        joined_str = str.join("", [a, b])
        final_int = int(joined_str)
        return final_int

    # n_before_c("127_C")
    # n_before_c("127_N")
    # n_before_c("134")

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
        }
    )
    rec_run_search = re.compile(r"^(?P<recno>\d{5,})_(?P<runno>\d+)_(?P<searchno>\d+)")
    _match = out["basename"].str.extractall(rec_run_search).reset_index()
    assert (_match.match == 0).all()
    assert (_match.index == _match.level_0).all()
    _match = _match.drop(columns=["match", "level_0"])
    _rec_run_search_value = f"{_match.iloc[0].recno}_{_match.iloc[0].runno}_{_match.iloc[0].searchno}"
    out = pd.concat([out, _match], axis=1)

    # rename each of the TMT_1xx quant columns as as rec_run_search_label
    _renamer = dict()
    for x in out.columns:
        if x.startswith("TMT_"):
            _renamer[x] = f"{_rec_run_search_value}_{x}"
            #out = out.rename(columns={x: f"{x}_{out.iloc[0]['basename']}"})
    out = out.rename(columns=_renamer)

    return out
    # note here is where df_nr_mednorm was being produced previously

    # qual_cols = set(df.columns) - {*set(id_cols), "AApos", "Protein", "Description"}
    # qual_cols = tuple(qual_cols)
    # qual_cols = sorted(qual_cols, key=n_before_c)
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
        ~(df.GeneID.str.startswith("CON")) |
        ~(df.GeneID.str.startswith("cont_")) |
        ~(df.GeneID.str.startswith("con"))
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
    if p.exists():
        df = pd.read_table(p, dtype={"GeneID": "str", "psm_GeneID": "str"})
        logging.info(f"Loaded {p}: {df.shape}")

    else:  # try ispec.
        from bcmproteomics_ext import ispec

        rec, run, search = parse_rawname(p)
        exp = ispec.PSMs(rec, run, search, data_dir=data_dir)
        df = exp.df
        # df = df.sample(100)
        # df = exp.df.head(40)
        # df = check_for_dup_cols(df)
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

    fa = pd.DataFrame.from_dict(fasta_dict_from_file(fasta, 'specific'))
    fa_generic = pd.DataFrame.from_dict(fasta_dict_from_file(fasta, 'generic'))
    # import ipdb; ipdb.set_trace()
    _ix = fa.loc[ fa.geneid.isna() ].index
    _to_replace = fa_generic.loc[_ix].header
    fa.loc[ _ix, 'geneid'] = _to_replace


    _ix = fa.loc[ fa.ref.isna() ].index
    _to_replace = fa_generic.loc[_ix].header
    fa.loc[ _ix, 'ref'] = _to_replace
    fasta_df = fa
    # import ipdb; ipdb.set_trace()

    for p in psms:
        logging.info(p)
        df = load(p, data_dir=data_dir)
        df = preprocess(df)
        geneids = df.GeneID.unique()
        # geneids = list(geneids)[0:100]
        df = df[ df.GeneID.isin(geneids) ]
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
                for res in pool.imap_unordered(
                    runner_partial, geneids #cores
                ):
                    # for res in pool.imap_unordered(runner, geneids=geneid, df=df, fa=fa, basename=basename, make_plot=plot, combine=combine, cores=cores, chunksize=1e2):
                    ALL_RESULTS.append(res)
                    pbar.update()
                # ALL_RESULTS = pool.imap_unordered(runner_partial, geneid[:510], cores)
                ALL_RESULTS = [
                    x for y in ALL_RESULTS for x in y
                ]  # unpack list of lists
        else:  # cores == 1
            ALL_RESULTS = runner(
                    geneids=geneids,
                df=df,
                fasta=fasta_df,
                basename=basename,
            )

        dfall = pd.concat(ALL_RESULTS)
        id_cols = [x for x in dfall.columns if x not in ("quantity", "label")]
        assert (dfall.groupby(id_cols).label.value_counts() == 1).all()
        # id_cols = [x for x in dfall.columns if x not in ("quantity", "label", "quality")]
        _dfall = dfall.pivot(index=id_cols, columns="label", values="quantity").reset_index()
        dfall = _dfall

        # now match to primary / secondary isoform and position
        # two new columns
        # column 1 is site type: primary, secondary, non selected unique, and non selected best
        # column 2 is site name: example EGFR_pY922
        # column 3 is the main, chosen, protein accession
        # column 4 is the main selected position
        #
        def add_prioritize_site_table(dfall, ref):
            res = pd.merge(dfall,
                ref[["Primary_select", "Secondary_select", 'Protein_id']],
                left_on="Protein", right_on="Protein_id", how="left"
             )
            return res

        _path = os.path.dirname(os.path.realpath(__file__))
        _f = os.path.join(_path, "GENCODE.V42.basic.CHR.isoform.selection.mapping.txt")
        _df = pd.read_csv(_f, sep="\t")
        dfall_with_priority_site = add_prioritize_site_table(dfall, _df)
        dfall = dfall_with_priority_site

        outname = os.path.join(data_dir, f"{basename}_site_table.tsv")

        # pivot here
        logging.info(f"Writing {outname}")
        dfall.to_csv(outname, sep="\t", index=False, mode="w")
        #

        #

        logging.info("making non redundant table")
        df_nr = make_nr(dfall)

        # it may be better to use a dictionary to map dfall[['AApos']] to df_nr
        # or do this:
        _mapping = dfall.drop_duplicates(['Protein', "protein_length"])
        _df_nr = pd.merge(
            df_nr,
            _mapping,
            left_on=["Final_select"],
            right_on=["Protein"],
            how="left",
        )
        _df_nr = _df_nr.drop(columns=["Protein"])
        df_nr = _df_nr


        outname = os.path.join(data_dir, f"{basename}_site_table_nr.tsv")
        df_nr.to_csv(outname, sep="\t", index=False, mode="w")
        logging.info(f"wrote {outname}")

        # _idx = ["recno", "runno", "searchno", "label"]
        # median_values = df_nr.groupby(_idx).quantity.transform("median")
        # df_nr_mednorm = df_nr.copy()
        # df_nr_mednorm['quantity'] = df_nr_mednorm['quantity'] / median_values
        # # med norm
        # outname = os.path.join(data_dir, f"{basename}_site_table_nr_mednorm.tsv")
        # df_nr_mednorm.to_csv(outname, sep="\t", index=False, mode="w")
        # logging.info(f"wrote {outname}")


        outname = os.path.join(data_dir, f"{basename}_site_table_GENCODE.V42.basic.CHR.isoform.selection.mapping.txt.tsv")
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
