import re
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
import pandas as pd
from typing import Collection, List

from .containers import RunContainer


def find_rec_run(target):
    "Try to get record, run, and search numbers with regex of a target string with pattern \d+_\d+_\d+"

    _, target = os.path.split(target)  # ensure just searching on filename
    rec_run_search = re.compile(r"^(\d+)_(\d+)_")

    match = rec_run_search.search(target)
    if match:
        recno, runno = match.groups()
        return recno, runno
    return


class PSM_Concat:
    """combine multiple fractions of psm files for a given experiment"""

    NAME = "PSM-Concat"

    def run(
        self,
        runcontainers: List[RunContainer] = None,
        outdir: Path = None,
        **kwargs,
    ) -> str:

        if runcontainers is None:
            raise ValueError("No input")

        filegroups = defaultdict(list)
        # for f in sorted(files):
        for container in runcontainers:
            mspcfile = container.get_file("MSPCRunner")
            if mspcfile is None:
                continue

            recrun = find_rec_run(container.stem)
            # print(recrun)
            if not recrun:
                recrun = container.stem[:10]
                print(f"Could not get group for {container}, using {recrun}")
                # continue
            if recrun:
                group = f"{recrun[0]}_{recrun[1]}"
                filegroups[group].append(mspcfile)

        for group, files in filegroups.items():
            print(group)
            for f in sorted(files):
                print(f)
            print(len(files))
            df = pd.concat(pd.read_table(f) for f in files)
            outname = f"{group}_6_psms_all.txt"
            df.to_csv(outname, sep="\t", index=False)
            print(f"Wrote {outname}")

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
