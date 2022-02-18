import os
import re
import warnings
from collections import defaultdict
from copy import deepcopy as copy
from functools import partial
from typing import Iterable, Tuple
import click

from annotate_protein import *
import click
import pandas as pd
from RefProtDB.utils import fasta_dict_from_file

@click.command()
@click.option("--all-genes", is_flag=True, default=False)
@click.option("--cores", default=1, show_default=True)
@click.option("--combine", is_flag=True, default=False)
# @click.option('-p', '--psms', type=click.Path(exists=True, dir_okay=False), multiple=True)
@click.option("-p", "--psms", type=str, multiple=True)
@click.option("-g", "--geneid", multiple=True)
@click.option("--plot/--noplot", is_flag=True)
@click.option("-o", "--out", default=None)
@click.option("--data-dir", default=".")
@click.option(
    "-f", "--fasta", type=click.Path(exists=True, dir_okay=False), help="fasta file"
)
def main(all_genes, cores, combine, psms, geneid, plot, out, fasta, data_dir="."):

    fa = None  # lazy load later
    for p in psms:
        # get file
        if os.path.exists(p):
            df = pd.read_table(p, dtype={"GeneID": "str", "psm_GeneID": "str"})
        else:  # try ispec.
            from bcmproteomics_ext import ispec

            rec, run, search = parse_rawname(p)
            # df = exp.df.head(30)
            # import ipdb

            # ipdb.set_trace()
            exp = ispec.PSMs(rec, run, search, data_dir=data_dir)
            df = exp.df

        # ============================
        df = df.rename(
            columns={
                k: k.split("psm_")[1]
                for k in [col for col in df.columns if col.startswith("psm_")]
            },
            # inplace=True,
        )

        df = df.rename(
            columns={"Modifications_abbrev": "Modifications", "Peptide": "Sequence"}
        )
        df = maybe_split_on_proteinid(df)
        df = remove_contaminants(df)
        df = wide_to_long_xinpei_table(df)
        if "PSM_UseFLAG" not in df:
            df["PSM_UseFLAG"] = 1
        if "IonScore" not in df:
            df["IonScore"] = -1

        df["GeneID"] = df["GeneID"].astype(str)
        df["Modifications"] = df["Modifications"].astype(str).fillna("")

        if all_genes:
            geneid = df.GeneID.unique()

        if out is None:
            basename = os.path.split(os.path.splitext(p)[0])[1]
            # basename = '{}_{}_{}'.format(os.path.split(os.path.splitext(p)[0])[1], g, label)
        else:
            basename = f"{out}"

        # now load fasta
        if fa is None:

            # fa = pd.DataFrame.from_dict(fasta_dict_from_file(fasta))
            # this is for uniprot
            fa = pd.DataFrame.from_dict(fasta_dict_from_file(fasta, "generic"))
            fa["geneid"] = fa.header
            fa["ref"] = fa.header

        # ====
        if cores > 1:

            # arglist = [[g, df, fa, basename, plot, combine]
            #            for g in np.array_split(geneid, cores)
            #  ]



            import runner
            # need to import the function from a different file for successful pickle
            runner_partial = partial(
                runner.runner, df=df, fa=fa, basename=basename, make_plot=plot, combine=combine
            )

            from multiprocessing import Pool, set_start_method

            try:
                set_start_method("spawn")
            except RuntimeError:
                pass

            ALL_RESULTS = list()
            # with Pool(processes=cores) as pool, logging_redirect_tqdm():

            with Pool(processes=cores) as pool:
                # ALL_RESULTS = pool.map(run, arglist)
                # geneid = geneid[:5]
                # for res in tqdm.tqdm(
                chunksize = min(len(geneid)//cores, 100) # what is a good number for this?
                print(f"Using chunksize {chunksize}")

                import enlighten

                pbar = enlighten.Counter(total=len(geneid), desc="pbar", unit="tick")
                # for res in progressbar.progressbar(
                #     pool.imap_unordered(runner_partial, geneid, cores),
                #     total=len(geneid),
                #     file=sys.stderr,
                #     redirect_stdout=True
                #     # file=orig_stdout,
                # ):
                for res in pool.imap_unordered(runner_partial, geneid, chunksize):
                # for res in pool.imap_unordered(runner, geneids=geneid, df=df, fa=fa, basename=basename, make_plot=plot, combine=combine, cores=cores, chunksize=1e2):
                    ALL_RESULTS.append(res)
                    pbar.update()
                # ALL_RESULTS = pool.imap_unordered(runner_partial, geneid[:510], cores)
                ALL_RESULTS = [
                    x for y in ALL_RESULTS for x in y
                ]  # unpack list of lists
        else:
            ALL_RESULTS = runner(
                geneid, df, fa, basename, make_plot=plot, combine=combine
            )

        if ALL_RESULTS and basename is not None and combine:
            _out = out
            if out is None:
                _out = os.path.basename(p)
            dfall = pd.concat(ALL_RESULTS)
            values_col = "quantity"
            name_col = "label"

            # now pivot for label
            dfall = dfall.pivot(
                index=[x for x in dfall if x not in (values_col, name_col)],
                values=values_col,
                columns=name_col,
            ).reset_index()
            outname = os.path.join(data_dir, f"{_out}_site_table.tsv")
            _out = None  # reset

            print(f"Writing {outname}")

            dfall.to_csv(outname, sep="\t", index=False, mode="w")


def runner(geneids, df, fa, basename, make_plot, combine):
    # print(os.getpid())
    # print(geneids, make_plot)
    if isinstance(geneids, str) or not isinstance(geneids, Iterable):
        geneids = (geneids,)
    # if not isinstance(geneid, Iterable)
    # print(len(geneids))
    # return 1
    ALL_RESULTS = list()
    print(f"total geneids: {len(geneids)}")
    for ix, g in enumerate(geneids):
        for label in df.LabelFLAG.unique():
            samplename = f"{basename}_{g}"
            res = annotate_protein(
                df[df.LabelFLAG == label],
                g,
                fa,
                basename=samplename,
                label=label,
                make_plot=make_plot,
                combine=combine,
            )
            if res:
                ALL_RESULTS.append(pd.concat(res))
        if ix % 100 == 0:
            print(ix)
    return ALL_RESULTS


if __name__ == "__main__":

    # from bcmproteomics_ext import ispec
    main()

    # psms = pd.read_table('../data/40677_1_1_none_0_psms.tab')
    # psms.rename(columns={x: x.strip('psm_') for x in psms}, inplace=True)

    # psms[['Sequence', 'Modification', 'SequenceModi']]

    # fa = load_fa()
    # geneid = 5241
    # # annotate_protein(psms, geneid, fa)
