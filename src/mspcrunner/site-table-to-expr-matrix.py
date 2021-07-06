#! /usr/bin/env python

import sys
import os
from pathlib import Path
import pandas as pd

from mokapot import read_pin


def main(f):
    df = pd.read_table(f)

    # tmp = df.head()['basename'].str.extract('(TMT_\d+[_\s]?)')
    tmp = df["basename"].str.extract("(TMT_\d+(_[N|C])?)")
    df["label"] = tmp[0]

    dfw = df.pivot(
        index=["Site", "AApos", "AA", "Modi", "GeneID", "Protein"],
        values="quantity",
        columns="label",
    ).reset_index()

    outname = f"{f.stem}_expression_matrix.tsv"
    print(f"Writing {outname}")
    dfw.to_csv(outname, sep="\t", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(f"Usage {__file__} path")
        sys.exit(0)

    if os.path.isfile(sys.argv[1]):
        files = (Path(sys.argv[1]),)
    else:
        files = Path(sys.argv[1]).glob("*site_table.tsv")
    for entry in files:
        main(entry)
