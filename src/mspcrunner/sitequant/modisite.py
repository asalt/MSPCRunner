import logging
import pandas as pd
from collections import defaultdict
from copy import deepcopy as copy
import re
import ipdb

import numpy as np


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


def aggregate_to_site(pept_df, sequence):

    sequence = "_" * 7 + sequence + "_" * 7  # easy solution??
    # don't forget to add 15

    grps = pept_df.groupby("AAindex")

    # grps.aggregate({'Site' : lambda x: sequence[x.index-(15+15):x.index+(15+15)]})

    # sequence[]})

    d = {}
    for AAix, psms in grps:
        # import ipdb; ipdb.set_trace()
        i = int(AAix)
        site = sequence[
            i + (-7 + 7) : i + (7 + 7 + 1)
        ]  # 7 before and 7 after, then add 7 because we padded sequence with _
        site = list(site)
        site[7] = site[7].lower()
        # it looks silly but easier to keep track of what happened
        d[i] = "".join(site)

    pept_df["Site"] = pept_df.AAindex.map(d)

    return pept_df


def modisite_quant(
    psms: pd.DataFrame,
    geneid: str,
    fa: pd.DataFrame,
    basename="basename",
    label="TMTxx",
):
    """
    it works
    psms needs these columns:
    GeneID

    fa needs these columns:
    GeneID
    """

    psms = psms[(psms.GeneID == str(geneid))].query("PSM_UseFLAG==1")
    psms = psms[psms.LabelFLAG == label]
    seqs = fa[fa.geneid == str(geneid)]  # will loop over each
    if psms.empty:
        # then did not pass PSM_UseFLAG==1. Can happen if an unlabeled peptide shows up in a tmt experiment
        # print("{} not in dataset".format(geneid))
        return
    ALL_RESULTS = list()
    # seqs is a pd.DataFrame
    # entry is a pd.Series, 1 row from seqs, 1 fasta entry
    # here is an example:
    # header                                     sp|Q9NVM9|INT13_HUMAN
    # description    Integrator complex subunit 13 OS=Homo sapiens ...
    # sequence       MKIFSESHKTVFVVDHCPYMAESCRQHVEFDMLVKNRTQGIIPLAP...
    # geneid                                     sp|Q9NVM9|INT13_HUMAN
    # ref                                        sp|Q9NVM9|INT13_HUMAN

    logging.info(f"Iterating through {len(seqs)} sequences")
    for ix, entry in seqs.iterrows():
        peptide_positions = get_positions(psms.Sequence, protein=entry.sequence)
        if not peptide_positions:
            continue

        peptides = make_peptides(peptide_positions, psms)
        pept_df = make_peptide_df(peptides)
        pept_df = aggregate_to_site(pept_df, entry.sequence)

        # _cols = ['Site', 'AApos', 'AA', 'Modi', 'quantity_without_modi']
        _cols = [
            "Site",
            "AApos",
            "AA",
            "Modi",
        ]
        # import ipdb; ipdb.set_trace()
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
        if "ref" in entry.index:
            logging.debug("ref in entry.index")
            protein = entry.ref
        if "ENSP" in entry.index:
            logging.debug("ENSP in entry.index")
            protein = entry.ENSP
            logging.debug("using ENSP")
        site_table = site_table.assign(
            GeneID=geneid,
            Symbol=entry.symbol,
            Protein=protein,
            Description=entry.description,
            protein_length=len(entry.sequence),
        ).sort_values(by="AApos")
        # print(entry.ref)
        # site_table['enrichvalue'] = (site_table.quantity - site_table.quantity_without_modi) / (site_table.quantity + site_table.quantity_without_modi)
        site_table["AApos"] = site_table["AApos"].astype(int)
        site_table["basename"] = basename
        site_table["label"] = label
        ALL_RESULTS.append(site_table)

    return ALL_RESULTS


def make_peptide_df(peptides):
    #
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
            # if not v.startswith("79.9") and not v.startswith("15.9949"):
            #     import ipdb; ipdb.set_trace()
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
    pept_df["AApos"] = pept_df.start + pept_df.Rel_ModiPos + 1  # add 1 for 1 indexing

    return pept_df

    # do_something(protein_seq=entry.sequence, peptide_positions=peptide_positions)

    # don't wrap at any of these indices


def make_peptides(peptide_positions, psms: pd.DataFrame, **kws):
    """
    peptide_positions is defaultdict(list)
    keys are peptide sequences (str)
    values are positions of modification
    """
    logging.info("making peptides")

    peptides = list()
    # import ipdb; ipdb.set_trace()
    # if TARGET in in
    for peptide_seq, positions in peptide_positions.items():
        for (start, end) in positions:
            # get all PEPTIDES that are of this sequence
            # not sure why drop duplicates seqmodi
            query = psms[psms.Sequence == peptide_seq].drop_duplicates(["SequenceModi"])

            for modi in query.Modifications.fillna(""):

                # peptide_seq_modi = query.SequenceModi.iloc[0]
                # import ipdb; ipdb.set_trace()
                for peptide_seq_modi in query.SequenceModi.unique():
                    # import ipdb; ipdb.set_trace()

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

                    pept_area = psms[psms.SequenceModi == peptide_seq_modi][
                        # "PrecursorArea_dstrAdj"
                        "SequenceArea"
                    ].sum()

                    best_score = psms[psms.SequenceModi == peptide_seq_modi][
                        "IonScore"
                    ].max()
                    # get peptide area for unmodified peptide

                    all_psm_records = psms[
                        (psms.Sequence.str.lower() == peptide_seq.lower())
                    ]

                    # if gene_psms[gene_psms.Sequence.str.lower() == "gilaadesvgtmgnr"].any() is True:
                    # gilaadesvgtmgnr
                    ## This check to see if any combination of every single modi is present
                    ## from https://stackoverflow.com/questions/3041320/regex-and-operator (?=.*foo)(?=.baz)
                    # _regex = ''.join(['(?=.*{})'.format(x) for x in modkeys])
                    # without_modi = gene_psms[ ~gene_psms.Modifications.str.contains(_regex) ]
                    ## but we do not want this

                    # without_modi = gene_psms[ ~gene_psms.Modifications.str.contains() ]
                    # if peptide_seq_modi == "G(TMT)ILAADES(Phospho)VGTMGNR":

                    p = Peptide(
                        peptide_seq,  # string
                        start,
                        end,
                        mod_dict=mod_dict,  # {int: "<modi>"}
                        quantity=pept_area,  # float
                        quality=best_score,  # float
                    )
                    peptides.append(p)

    peptides = sorted(peptides, key=lambda x: x.start)
    return peptides
