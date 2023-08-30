import pandas as pd
from modisite import modisite_quant
import logging
from typing import Iterable, Tuple


def runner(
    geneids=None,
    df: pd.DataFrame = None,
    fasta: pd.DataFrame = None,
    basename="<basename>",
):
    # geneids = df["GeneID"]
    # geneids = geneids[:4]
    # print(os.getpid())
    # print(geneids, make_plot)
    if geneids is None:
        geneids = df.GeneID.unique()
    if isinstance(geneids, str) or not isinstance(geneids, Iterable):
        # print('----')
        geneids = [
            geneids,
        ]
    df = df[df.GeneID.isin(geneids)]
    # import ipdb; ipdb.set_trace()
    ALL_RESULTS = list()
    # print(f"total geneids: {len(geneids)}")
    for ix, g in enumerate(geneids):
        for label in df.LabelFLAG.unique():
            samplename = f"{basename}_{g}"
            res = modisite_quant(
                df[df.LabelFLAG == label],
                g,
                fasta,
                basename=samplename,
                label=label,
            )
            if res is None or len(res) == 0:
                if "cont" in g.lower():  # this is an exception we can skip
                    continue
                elif "cont" not in g.lower():

                    logging.warning("res is None or empty")
                    logging.warning(
                        f"""
                    geneid: {g}
                    label: {label}
                    basename: {samplename}
                    """
                    )
                    # import ipdb;ipdb.set_trace()
                    continue
            _res_df = pd.concat(res)
            ALL_RESULTS.append(_res_df)
        # if ix % 100 == 0:
        #     print(ix)
    return ALL_RESULTS
