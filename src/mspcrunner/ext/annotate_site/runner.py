
import pandas as pd
from annotate_protein import annotate_protein
from typing import Iterable, Tuple
def runner(geneids, df, fa, basename, make_plot, combine):
    #import ipdb; ipdb.set_trace()
    # print(os.getpid())
    # print(geneids, make_plot)
    if isinstance(geneids, str) or not isinstance(geneids, Iterable):
        geneids = (geneids,)
    # import ipdb; ipdb.set_trace()
    ALL_RESULTS = list()
    # print(f"total geneids: {len(geneids)}")
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
        # if ix % 100 == 0:
        #     print(ix)
    return ALL_RESULTS
