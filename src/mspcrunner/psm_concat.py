from ipdb import set_trace
import re
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
import pandas as pd
from typing import Collection, List

from .containers import RunContainer, SampleRunContainer
from .commands import Receiver

from .logger import get_logger

logger = get_logger(__name__)


def find_rec_run(target: str):
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
        containers: List[RunContainer] = None,
        outdir: Path = None,
        force=False,
        **kwargs,
    ) -> str:
        """
        We can use this procedure to create SampleRunContainers
        """

        if containers is None:
            logger.error(f"no runcontainers passed")
            # this is bad
            return
            # raise ValueError("No input")

        logger.debug(f"{self}")
        filegroups = defaultdict(list)
        # for f in sorted(files):
        for container in containers:
            if not isinstance(container, RunContainer):
                continue  # wrong container
            mspcfile = container.get_file("MSPCRunner")
            if mspcfile is None:
                logger.debug(f"No MSPCRunner file for {container}")
                continue

            # this is where search could be designated
            recrun = find_rec_run(container.stem)
            # print(recrun)
            if not recrun:
                recrun = container.stem[:10]
                logger.warn(f"Could not get group for {container}, using {recrun}")
                # continue

            if recrun:
                group = f"{recrun[0]}_{recrun[1]}"
                # populate all of our "filegroups"
                # filegroups[group].append(mspcfile)
                filegroups[group].append(container)

        sampleruncontainers = list()

        # create run containers

        for group, runcontainers in filegroups.items():

            # recrun = find_rec_run(container.stem)
            # if not recrun:  # ..
            #     continue

            recrun = {find_rec_run(container.stem) for container in runcontainers}
            rootdir = {container.rootdir for container in runcontainers}
            assert len(recrun) == 1
            recrun = list(recrun)[0]
            assert len(rootdir) == 1
            rootdir = list(rootdir)[0]

            record_no = recrun[0]
            run_no = recrun[1]
            rootdir = rootdir

            samplerun = SampleRunContainer(
                name=group,
                rootdir=rootdir,
                runcontainers=runcontainers,
                record_no=record_no,
                run_no=run_no,
            )

            sampleruncontainers.append(samplerun)
            print(group)

            # # move this?
            # for f in sorted(files):
            #     print(f)
            # print(len(files))
            # df = pd.concat(pd.read_table(f) for f in files)
            # outname = f"{group}_6_psms_all.txt"
            # df.to_csv(outname, sep="\t", index=False)
            # print(f"Wrote {outname}")

        for samplerun in sampleruncontainers:
            samplerun.check_psms_files()
            samplerun.concat(force=force)

        return sampleruncontainers


class SampleRunContainerBuilder(Receiver):
    """combine multiple fractions of psm files for a given experiment"""

    NAME = "PSM-Collect"

    def run(
        self,
        containers: List[RunContainer] = None,
        outdir: Path = None,
        **kwargs,
    ) -> str:
        """
        We can use this procedure to create SampleRunContainers
        """

        if containers is None:
            logger.error(f"no runcontainers passed")
            # this is bad
            return
            # raise ValueError("No input")

        # logger.debug(f"{self}")
        filegroups = defaultdict(list)
        # for f in sorted(files):
        for container in containers:
            if not isinstance(container, RunContainer):
                continue  # wrong container

            # this is where search could be designated
            recrun = find_rec_run(container.stem)
            # print(recrun)
            if not recrun:
                recrun = container.stem[:10]
                logger.warn(f"Could not get group for {container}, using {recrun}")
                # continue

            if recrun:
                group = f"{recrun[0]}_{recrun[1]}"
                # populate all of our "filegroups"
                # filegroups[group].append(mspcfile)
                filegroups[group].append(container)

        sampleruncontainers = list()

        # create run containers

        for group, runcontainers in filegroups.items():

            # recrun = find_rec_run(container.stem)
            # if not recrun:  # ..
            #     continue

            recrun = {find_rec_run(container.stem) for container in runcontainers}

            # silently drops RunContainers that do not have a pin file
            rootdir = filter(None, {container.rootdir for container in runcontainers})
            rootdir = list(rootdir)
            assert len(recrun) == 1
            recrun = list(recrun)[0]
            # import ipdb; ipdb.set_trace()
            assert len(rootdir) == 1
            rootdir = list(rootdir)[0]

            record_no = recrun[0]
            run_no = recrun[1]
            rootdir = rootdir

            samplerun = SampleRunContainer(
                name=group,
                rootdir=rootdir,
                runcontainers=runcontainers,
                record_no=record_no,
                run_no=run_no,
            )

            sampleruncontainers.append(samplerun)
        return sampleruncontainers
        # print(group)

        # # move this?
        # for f in sorted(files):
        #     print(f)
        # print(len(files))
        # df = pd.concat(pd.read_table(f) for f in files)
        # outname = f"{group}_6_psms_all.txt"
        # df.to_csv(outname, sep="\t", index=False)
        # print(f"Wrote {outname}")

        # for samplerun in sampleruncontainers:
        #     samplerun.check_psms_files()
        #     samplerun.concat(force=force)

        # return sampleruncontainers


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
