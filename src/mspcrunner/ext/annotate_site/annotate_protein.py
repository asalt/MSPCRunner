import os
import re
import warnings
from collections import defaultdict
from copy import deepcopy as copy
from functools import partial
from typing import Iterable, Tuple
import ipdb

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams["text.usetex"] = False
# https://github.com/tqdm/tqdm#redirecting-logging
import contextlib

# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import sys

import numpy as np
import seaborn as sb
import tqdm
from tqdm.contrib import DummyTqdmFile
from tqdm.contrib.logging import logging_redirect_tqdm

import click
import pandas as pd
from RefProtDB.utils import fasta_dict_from_file


def parse_rawname(name: str) -> Tuple[str, str, str]:
    """yield up to the first 3 numbers in a string separated by underscore
    returns None when number is missing / interrupted
    """
    namesplit = name.split("_")
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


# mpl.use('ps')
# mpl.use('pgf')
# mpl.use('pdf')
# mpl.rc('text', usetex=False)
# mpl.rc('font.family', 'monospace')
def maybe_split_on_proteinid(df):
    # look at geneid column
    SEP = ";"
    if "Proteins" not in df:
        return df
    if not df["Proteins"].str.contains(SEP).any():
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
        ~(df.GeneID.str.startswith("CON_")) & ~(df.GeneID.str.startswith("cont_"))
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


class AA:
    def __init__(self, a, pos):
        if len(a) > 1:
            raise ValueError("AA must be length 1")
        self.a = a
        self.pos = pos

    def __repr__(self):
        return self.a

    def __str__(self):
        return self.a

    def __lt__(self, other):
        return self.pos < other.pos

    def __gt__(self, other):
        return self.pos > other.pos

    def __le__(self, other):
        return self.pos <= other.pos

    def __ge__(self, other):
        return self.pos >= other.pos


class Peptide:
    def __init__(self, seq, start, end, mod_dict=None, quantity=0, quality=None):
        self.seq = seq
        self.start = start
        self.end = end
        self.row_position = None  # to be assigned later, or not?
        self.mod_dict = mod_dict  # optional dictionary of positional AA modifications
        self.quantity = quantity
        self.quality = quality

    def __repr__(self):
        return "Peptide ({},{}) : {}".format(self.start, self.end, self.seq)

    def __str__(self):
        return self.seq

    def __len__(self):
        return len(self.seq)

    def __lt__(self, other):
        return self.end < other.start

    def __gt__(self, other):
        return self.start > other.end


class Row:
    def __init__(self, rowpos):
        if not isinstance(rowpos, int):
            raise ValueError("Must be integer")
        self.rowpos = rowpos
        self.sequence_list = list()
        self._sequence = None
        self.peptides = list()
        self._peptide_span = None

    @property
    def sequence(self):
        if self._sequence is None or len(self._sequence) < len(self.sequence_list):
            # TODO: ensure proper sorted order
            self._sequence = "".join(str(x) for x in self.sequence_list)
        return self._sequence

    @property
    def peptide_span(self):
        if self._peptide_span is None or len(
            [x for row in self._peptide_span.values() for x in row]
        ) < len(self.peptides):
            self.place_peptides()
        return self._peptide_span

    def place_peptides(self):

        # from copy import copy
        # peptides = copy(self.peptides)
        from collections import deque

        peptides = deque(self.peptides)

        rows = defaultdict(list)

        ix = 0
        while peptides:
            row = rows[ix]
            for pep_ix in range(len(peptides)):
                # print(ix, row, pep_ix)
                if len(peptides) == 0:
                    ix += 1
                    break  # done
                if not row:  # new/empty row, add peptide
                    rows[ix].append(peptides.popleft())
                    continue
                pept = peptides[0]
                if any(
                    (pept.start >= p.start and pept.start < p.end)
                    or (pept.end > p.start and pept.end <= p.end)
                    for p in row
                ):
                    continue  # overlap
                else:
                    rows[ix].append(peptides.popleft())
            else:
                ix += 1

        self._peptide_span = rows

    def __len__(self):
        return len(self._sequence)

    def append(self, AA):
        self.sequence_list.append(AA)

    def append_multiple(self, AAs):
        for AA in AAs:
            self.append(AA)

    def add_peptide(self, peptide):

        seq = self.sequence
        # in case there are more than one
        positions = get_positions([peptide.seq], seq)[peptide.seq]
        # shift = self.sequence_list[0].pos
        if not positions:  # not in here
            raise ValueError("Could not find {} in {}".format(peptide, self))
        for (start, end) in positions:
            p = Peptide(peptide.seq, start, end, mod_dict=peptide.mod_dict)
        self.peptides.append(p)


# TODO expand, absolute path..
FA = {"hs": "../data/RefSeq/gpGrouper_9606_refseq_2019_01_14.fa"}

# lru cache?
# def load_fa(species="hs"):
#     if species not in FA:
#         raise ValueError("`species` must be one of : {}".format(" ".join(FA.keys())))
#
#     f = FA[species]
#     fa = pd.DataFrame.from_dict(fasta_dict_from_file(f))
#     return fa
#


def get_positions(peptides, protein):

    """
    returns a dict of lists of start,end positions for each peptide
    in given protein.
    """

    res = defaultdict(list)

    for p in set(peptides):
        pos = 0
        start = 0
        while pos != -1:

            pos = protein.find(p.upper(), start)

            if pos == -1:
                continue  # not in here

            endpos = pos + len(p)

            res[p].append((pos, endpos))

            start = endpos

    return res


# not this
# from biograpy import drawer, tracks, features
# from biograpy.features import Simple
# from biograpy.tracks import BaseTrack
# from biograpy.drawer import Panel


label_pos = re.compile(
    """
    (?<!Label\:)  # not preceded by
    (\d+|N-Term)      # Match numbers,
    (?!       # only if it's not followed by..."
     [^(]*    #   any number of characters except opening parens" +
     \\)      #   followed by a closing parens" +
    )         # End of lookahead""",
    re.VERBOSE,
)
inside_paren = re.compile(r"\((\S+)\)")


def annotate_protein(
    psms, geneid, fa, basename=None, label=None, make_plot=True, combine=False
):

    # print(basename, make_plot)

    # rc = {"pgf.rcfonts": False,           # Ignore Matplotlibrc
    #       "text.usetex": True,            # use LaTeX to write all text do we need this?
    #       "font.family": 'monospace',
    #       "pgf.rcfonts": False,           # Ignore Matplotlibrc
    #       "pgf.preamble": [
    #           r'\usepackage[dvipsnames]{xcolor}'     # xcolor for colours
    #           # r'\usepackage{inconsolata}',
    #           # r'\usepackage[letterpaper, landscape, margin=.5in]{geometry}'
    #       ]
    # }

    gene_psms = psms[(psms.GeneID == str(geneid))].query("PSM_UseFLAG==1")
    seqs = fa[fa.geneid == str(geneid)]  # will loop over each
    if gene_psms.empty:
        # then did not pass PSM_UseFLAG==1. Can happen if an unlabeled peptide shows up in a tmt experiment
        # print("{} not in dataset".format(geneid))
        return

    ALL_RESULTS = list()
    for ix, entry in seqs.iterrows():

        outname = f"{basename}_{label}_{entry.ref}.pdf"
        protein_seq = entry.sequence
        peptide_positions = get_positions(gene_psms.Sequence, protein_seq)

        if not peptide_positions:
            continue

        # don't wrap at any of these indices
        # np.arange from x to y-1 because it is OK to break at the final amino acid
        # of a peptide sequence
        protected_regions = set(
            np.concatenate(
                [
                    np.arange(x, y - 1)
                    for (x, y) in [j for i in peptide_positions.values() for j in i]
                ]
            )
        )

        # find the max length of a row
        rows = defaultdict(list)
        row = 0
        counter = 0
        for ix, s in enumerate(protein_seq):
            aa = AA(s, ix)
            rows[row].append(aa)
            counter += 1
            if counter > 80 and ix not in protected_regions:
                row += 1
                counter = 0
        # import ipdb; ipdb.set_trace()

        max_row_len = max(len(x) for x in rows.values())

        # make all Peptides
        peptides = list()
        for peptide_seq, positions in peptide_positions.items():
            for (start, end) in positions:
                query = gene_psms[gene_psms.Sequence == peptide_seq].drop_duplicates(
                    ["SequenceModi"]
                )
                for modi in query.Modifications.fillna(""):

                    peptide_seq_modi = query.SequenceModi.iloc[0]

                    modkeys = inside_paren.findall(modi)
                    modpos = label_pos.findall(modi)
                    modpos = [
                        0 if x.lower() == "n-term" else int(x) - 1 for x in modpos
                    ]  # n terminal is at first position
                    _ignore = (
                        "229.16",
                        "304.207",
                        "57.02",
                        "TMT",
                    )  # dynamic tmt? or just in general. either way we ignore
                    mod_dict = {
                        a: b
                        for a, b in zip(modpos, modkeys)
                        if not any(x in b for x in _ignore)
                    }

                    pept_area = gene_psms[gene_psms.SequenceModi == peptide_seq_modi][
                        # "PrecursorArea_dstrAdj"
                        "SequenceArea"
                    ].sum()
                    # get peptide area for unmodified peptide

                    all_psm_records = gene_psms[
                        (gene_psms.Sequence.str.lower() == peptide_seq.lower())
                    ]
                    ## This check to see if any combination of every single modi is present
                    ## from https://stackoverflow.com/questions/3041320/regex-and-operator (?=.*foo)(?=.baz)
                    # _regex = ''.join(['(?=.*{})'.format(x) for x in modkeys])
                    # without_modi = gene_psms[ ~gene_psms.Modifications.str.contains(_regex) ]
                    ## but we do not want this

                    # without_modi = gene_psms[ ~gene_psms.Modifications.str.contains() ]

                    # DONE check on SequenceModi, not Sequence
                    best_score = gene_psms[gene_psms.SequenceModi == peptide_seq_modi][
                        "IonScore"
                    ].max()

                    p = Peptide(
                        peptide_seq,
                        start,
                        end,
                        mod_dict=mod_dict,
                        quantity=pept_area,
                        quality=best_score,
                    )
                    peptides.append(p)

        peptides = sorted(peptides, key=lambda x: x.start)

        ## TODO: put this in separate function please
        _fields = ["seq", "start", "end", "quantity", "quality"]
        ## build dict
        pept_list = list()
        for pept in peptides:
            # if pept.seq != "LAVGRHsFSR":
            #     continue
            d = dict()
            for a in _fields:
                d[a] = getattr(pept, a)

            d["all_modis"] = "|".join(x for x in set(pept.mod_dict.values()))

            if not pept.mod_dict:
                pept_list.append(d)
            for k, v in pept.mod_dict.items():
                d_copy = copy(d)
                d_copy["Rel_ModiPos"] = k
                d_copy["Modi"] = v
                d_copy["AA"] = pept.seq[k]
                pept_list.append(d_copy)

        # pept_dict = [[getattr(pept, a) for a in _fields] for pept in peptides]

        pept_df = pd.DataFrame(pept_list)

        ## if there are no modifications at all, need to make these empty columns
        for col in ["Rel_ModiPos", "Modi", "AA"]:
            if col not in pept_df:
                pept_df[col] = np.nan

        pept_df["AAindex"] = pept_df.start + pept_df.Rel_ModiPos
        pept_df["AApos"] = (
            pept_df.start + pept_df.Rel_ModiPos + 1
        )  # add 1 for 1 indexing

        # no don't do this
        # now get the quantity from all peptides that DO NOT contain a given modification
        # for downstream normalization

        # pept_df["Modi"] = pept_df["Modi"].fillna("")
        # unmodified_sites = dict()
        # for ix, grp in pept_df.groupby(["start", "Modi"]):
        #     start, modi = ix
        #     # if start == 548: import ipdb; ipdb.set_trace()
        #     tot = pept_df[
        #         (pept_df.start == start) & (~pept_df.all_modis.str.contains(modi))
        #     ].quantity.sum()
        #     unmodified_sites[(start, modi)] = tot

        # unmodified_sites_df = pd.DataFrame(
        #     data=unmodified_sites.values(), index=unmodified_sites.keys()
        # ).reset_index()
        # unmodified_sites_df.columns = ["start", "Modi", "quantity_without_modi"]
        # pept_df = pept_df.merge(unmodified_sites_df, on=["start", "Modi"])

        from modisite import aggregate_to_site

        pept_df = aggregate_to_site(pept_df, entry.sequence)

        # _cols = ['Site', 'AApos', 'AA', 'Modi', 'quantity_without_modi']
        _cols = [
            "Site",
            "AApos",
            "AA",
            "Modi",
        ]
        site_table = (
            pept_df.groupby("Site")
            .aggregate(
                {
                    "quantity": sum,
                    "quality": max,
                }
            )
            .reset_index()
            .merge(pept_df[_cols].drop_duplicates("Site"), on="Site", how="left")
        )
        site_table = site_table.assign(GeneID=geneid, Protein=entry.ref).sort_values(
            by="AApos"
        )
        # print(entry.ref)
        # site_table['enrichvalue'] = (site_table.quantity - site_table.quantity_without_modi) / (site_table.quantity + site_table.quantity_without_modi)
        site_table["AApos"] = site_table["AApos"].astype(int)
        site_table["basename"] = basename
        site_table["label"] = label
        # if basename is not None and combine:
        if combine:
            ALL_RESULTS.append(site_table)
            # header = True
            # if os.path.exists(basename+'_site_table.tsv'):
            #    header=False
            # .sort_values(by='AApos').to_csv(basename+'_site_table.tsv', sep='\t', index=False, mode='a+', header=header) else:

            # ??
            # site_table.sort_values(by="AApos").to_csv(
            #     outname + "_site_table.tsv", sep="\t", index=False, mode="w+"
            # )

        if not make_plot:
            continue

        rows_withpepts = list()
        for row_ix, aas in rows.items():
            row = Row(row_ix)
            row.append_multiple(aas)
            firstpos = aas[0].pos
            lastpos = aas[-1].pos
            row_peptides = [
                p for p in peptides if (p.start >= firstpos and p.end <= lastpos + 1)
            ]
            for p in row_peptides:
                try:
                    row.add_peptide(p)
                except ValueError:  # if gene not in dataset
                    pass
            rows_withpepts.append(row)

        all_modis = set(m for p in peptides for m in p.mod_dict.values())

        # print(basename, outname)
        title = outname
        fig = plot_func(rows_withpepts, modis=all_modis, outname=outname, title=title)
        # we're done now plot

        # fig.savefig(outname)
    return ALL_RESULTS


from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.ticker import MultipleLocator


def plot_func(rows, modis=None, title=None, outname=None):
    """rows is a collection of Row objects with peptides inside"""
    orig_rc = mpl.rcParams

    mpl.rcParams["font.family"] = "monospace"
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["text.usetex"] = False

    DEFAULTS = {
        "Oxidation": "red",
        "Methyl": "yellow",
        "Phospho": "green",
        "Carbamidomethyl": "grey",
        "GlyGly": "purple",
    }
    modi_colors = dict(DEFAULTS)

    if modis:
        colors = iter([mpl.colors.rgb2hex(x) for x in sb.color_palette()])
        for m in modis:
            if m in modi_colors:
                continue
            modi_colors[m] = next(colors)

    # total row size
    nrows = 0
    for row in rows:
        nrows += 1 + len(row.peptide_span)

    max_len = max(len(row.sequence) for row in rows)

    # 1/4 inch per row?
    figlen = max(nrows * 0.25, 6)
    # length
    figwidth = max_len * 7 / 80

    figsize = (figwidth, figlen)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_yticks(np.arange(0, nrows, 1))
    ax.set_xticks(np.arange(0, max_len, 1))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(which="major", axis="x", color="#CCCCCC", linestyle="--", linewidth=1)

    # get size of axes for drawing circles
    x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixes
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)
    modi_width = 0.5 * maxd / dx
    modi_height = 0.5 * maxd / dy

    rects = list()
    y = nrows
    for row in rows:

        ax.text(
            -2,
            y,
            row.sequence_list[0].pos,
            horizontalalignment="right",
            fontdict=dict(family="monospace", weight="normal", size=6),
        )
        # ax.text(ix, y, row.sequence, fontdict=dict(family='monospace', weight='bold', size=8))
        for ix, s in enumerate(row.sequence):
            ax.text(ix, y, s, fontdict=dict(family="monospace", weight="bold", size=8))

        for peptides in row.peptide_span.values():
            y -= 1
            for p in peptides:
                rect = Rectangle(
                    (p.start, y), width=len(p), height=0.75, facecolor="#bbbbbb"
                )
                ax.add_patch(rect)
                if p.mod_dict is not None:
                    for ix, mod in p.mod_dict.items():
                        # patch = Ellipse((p.start+ix+.5, y+.5), modi_width, modi_height,
                        #                 facecolor=modi_colors[mod], edgecolor='#222222')
                        patch = Line2D(
                            [p.start + ix + 0.5],
                            [y + 0.40],
                            marker="o",
                            markeredgewidth=0.5,
                            markerfacecolor=modi_colors[mod],
                            markeredgecolor="#222222",
                        )
                        ax.add_artist(patch)
                # rects.append(rect)

        y -= 1  # room for next row
        # y -= len(row.peptide_span)+1

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_yticklabels([])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    # ax.set_xticklabels([])

    if modi_colors is not None:
        handles, labels = list(), list()
        for k, c in modi_colors.items():
            handle = mpl.lines.Line2D(
                range(1),
                range(1),
                color="none",
                marker="o",
                markerfacecolor=c,
                markeredgewidth=0.5,
                markeredgecolor="#111111",
            )
            handles.append(handle)
            labels.append(k)
        leg = ax.legend(
            handles,
            labels,
            ncol=4,
            fontsize=8,
            bbox_to_anchor=(1, 0),
            loc="lower right",
            bbox_transform=fig.transFigure,
        )

    if title is not None:
        ax.set_title(title, fontsize=12, y=1.16)

    # fig.tight_layout(rect=(0,.05,1,1))
    fig.tight_layout()
    fig.savefig(outname)
    plt.close(fig)

    return fig

    # for pept, p in positions.items():
    #     score = gene_psms[ gene_psms.Sequence == pept].IonScore.max()
    #     for (l, r) in p:
    #         feat = Simple(name = pept, start=l+1, end=r+1, score=score)
    #         # feat = Simple(name = '', start=l+1, end=r+1)
    #         feats.append(feat)


def check_for_dup_cols(df):
    tot = len([x for x in df.columns if x == "Modifications"])
    if tot != 1:
        tmp = df.Modifications.values
        df = df.drop(columns="Modifications", axis=1)
        df["Modifications"] = tmp
    return df


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
            exp = ispec.PSMs(rec, run, search, data_dir=data_dir)
            df = exp.df
            # df = check_for_dup_cols(df)

        # ============================
        df["GeneID"] = df.GeneID.astype(str)
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
        df = maybe_split_on_proteinid(df)
        df = remove_contaminants(df)
        # special case
        # df = wide_to_long_xinpei_table(df)
        if "PSM_UseFLAG" not in df:
            df["PSM_UseFLAG"] = 1
        if "IonScore" not in df:
            df["IonScore"] = -1

        df["GeneID"] = df["GeneID"].astype(str)
        df["Modifications"] = df["Modifications"].astype(str).fillna("")

        if all_genes:
            if (
                "ReporterIntensity" in df.columns
                and not df.ReporterIntensity.isna().all()
            ):
                df = df[~df.ReporterIntensity.isna()]
            geneid = df.GeneID.unique()

        if out is None:
            basename = os.path.split(os.path.splitext(p)[0])[1]
            # basename = '{}_{}_{}'.format(os.path.split(os.path.splitext(p)[0])[1], g, label)
        else:
            basename = f"{out}"

        # now load fasta
        if fa is None:

            fa = pd.DataFrame.from_dict(fasta_dict_from_file(fasta))
            if "geneid" not in map(
                lambda x: x.lower(), fa.columns
            ):  # this is not a comprehensive check
                # this is for uniprot
                fa = pd.DataFrame.from_dict(fasta_dict_from_file(fasta, "generic"))
                fa["geneid"] = fa.header
                fa["ref"] = fa.header

        # ====
        from runner import runner

        if cores > 1:

            # arglist = [[g, df, fa, basename, plot, combine]
            #            for g in np.array_split(geneid, cores)
            #  ]

            runner_partial = partial(
                runner,
                df=df,
                fa=fa,
                basename=basename,
                make_plot=plot,
                combine=combine,
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
                # geneid = geneid[:42]
                # for res in tqdm.tqdm(

                import enlighten

                pbar = enlighten.Counter(total=len(geneid), desc="pbar", unit="tick")
                # for res in progressbar.progressbar(
                #     pool.imap_unordered(runner_partial, geneid, cores),
                #     total=len(geneid),
                #     file=sys.stderr,
                #     redirect_stdout=True
                #     # file=orig_stdout,
                # ):
                for res in pool.imap_unordered(runner_partial, geneid, cores):
                    # for res in pool.imap_unordered(runner, geneids=geneid, df=df, fa=fa, basename=basename, make_plot=plot, combine=combine, cores=cores, chunksize=1e2):
                    ALL_RESULTS.append(res)
                    pbar.update()
                # ALL_RESULTS = pool.imap_unordered(runner_partial, geneid[:510], cores)
                ALL_RESULTS = [
                    x for y in ALL_RESULTS for x in y
                ]  # unpack list of lists
        else:  # cores == 1
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
