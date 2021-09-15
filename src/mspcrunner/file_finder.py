from ipdb import set_trace
import re
from collections import defaultdict
from typing import List, Collection
from pathlib import Path


from .logger import get_logger
from .containers import AbstractContainer, RunContainer, SampleRunContainer

logger = get_logger(__name__)


class FileFinder:  # receiver
    NAME = "FileFinder Receiver"

    # PATTERNS = ["*raw", "*mzML", "*txt", "*pin", "*tsv"]
    # # use these to trim
    # FILE_EXTENSIONS = [
    #     ".mokapot.psms",
    #     ".mokapot.peptides",
    #     "_ReporterIons",
    #     "_SICstats",
    #     "_ScanStats",
    #     "_ScanStatsConstant",
    #     "_MSPCRunner_a1",
    # ]

    # RESULT_FILES = ["psms_all.txt", "e2g_QUAL.tsv", "e2g_QUANT.tsv"]

    def run(
        self,
        files=None,
        path: Path = None,
        container_obj: AbstractContainer = None,
        depth=5,
        **kws,
    ) -> Collection[RunContainer]:
        """
        Finds files and creates `contaner_obj: AbstractContainer`
        `container_obj` has an "add_file" method for collecting files it deems relevant.
        """

        assert isinstance(container_obj(), AbstractContainer)

        # we need the add_file method

        # if path is None:
        #     path = Path(".")

        # RESULT_FILES = self.RESULT_FILES
        # res = li()
        # logger.debug(f"file:{file} path:{path}")
        # res = defaultdict(RunContainer)
        res = defaultdict(container_obj)
        observed_files = list()

        if files is not None:  # not working yet
            for f in files:
                basename = f.stem

                # if basename.endswith("F1"):
                res[basename].add_file(f.resolve())

        if path is None:
            return res.values()
        # logger.debug([x for x in path.glob("*")])

        # IDEA turn into a factory?
        # return Container(**).construct()

        for pat in container_obj.PATTERNS:
            for i in range(depth):
                globstr = "*/" * i + pat
                # bug when depth == 1 and file has moved into its new home directory
                for f in path.glob(globstr):  # DONE: ? fix if path is None
                    # if (not f.is_file()) and (not f.is_symlink()):

                    # logger.debug(f"pat:{pat}, file:{f}")
                    if f.is_dir():
                        continue
                    if f.name in observed_files:
                        logger.debug(f"already have a file {f.name} skipping ")
                        continue

                    # print(f)
                    # recno, runno, searchno = parse_rawname(f.stem)
                    # if searchno is None:
                    #    searchno = "6"
                    # name=parse_rawname(f.name)
                    # full_name =  f"{recno}_{runno}_{searchno}"

                    # basename = f.stem
                    basename = container_obj.make_basename(f)
                    if basename is None:  # if is some irrelevant file
                        continue

                    # if "psms_all" in basename:
                    #     set_trace()

                    if f.is_symlink():
                        raise ValueError(f"No symlink allowed")

                    # sampleruncontainer "basename" is rec_run_search
                    if (
                        basename is not False
                        and isinstance(container_obj(), SampleRunContainer)
                        and "psms_all" in basename
                    ):
                        pass
                        # logger.debug(f)
                        # set_trace()
                        # 1 + 1

                    if basename is not None:
                        res[basename].add_file(f)
                        # the defaultdict collection made it convienent to keep adding
                        # files to the same "base" file

                    #     # add files to RunContainers with a matching `_name`
                    observed_files.append(f.name)
                    # weird behavior fix later
                    # run_container = RunContainer(stem=f.stem)
                    # res.append(run_container)

        # set_trace()
        if len(res) > 0:  # for debugging
            _key = list(res.keys())[0]
            _first = res[_key]
        ret = [x for x in res.values() if x.n_files > 0]
        import random

        random.shuffle(ret)

        # ret = {
        #     f"{self.NAME}": {
        #         str(container_obj): [x for x in res.values() if x.n_files > 0]
        #     }
        # }

        return ret
