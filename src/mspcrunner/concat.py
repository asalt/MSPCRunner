from containers import RunContainer
import re
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
import pandas as pd

from .utils import find_rec_run


class PSM_Concat:
    """combine multiple fractions of psm files for a given experiment"""

    NAME = "PSM-Concat"

    def run(
        self, runcontainer: RunContainer = None, outdir: Path = None, **kwargs
    ) -> str:

        if runcontainer is None:
            raise ValueError("No input")
        1 + 1
        return "s"


if __name__ == "__main__":

    files = glob("*MSPCRunner*.txt")
    filegroups = defaultdict(list)
    for f in sorted(files):
        recrun = find_rec_run(f)
        # print(recrun)
        if not recrun:
            print(f"Could not get group for f{f}")
            continue
        if recrun:
            group = f"{recrun[0]}_{recrun[1]}"
            filegroups[group].append(f)

    for group, files in filegroups.items():
        print(group)
        for f in files:
            print(f)
        print(len(files))
        df = pd.concat(pd.read_table(f) for f in files)
        outname = f"{group}_6_psms_all.txt"
        df.to_csv(outname, sep="\t", index=False)
        print(f"Wrote {outname}")
