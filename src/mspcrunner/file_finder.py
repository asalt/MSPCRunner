from collections import defaultdict
from typing import List, Collection
from pathlib import Path

import ipdb

from .logger import get_logger
from .containers import RunContainer

logger = get_logger(__name__)


class FileFinder:  # receiver
    NAME = "FileFinder Receiver"

    PATTERNS = ["*raw", "*mzML", "*txt", "*pin", "*tsv"]
    FILE_EXTENSIONS = [
        ".mokapot.psms",
        ".mokapot.peptides",
        "_ReporterIons",
        "_SICstats",
        "_ScanStats",
        "_ScanStatsConstant",
        "_MSPCRunner_a1",
    ]

    RESULT_FILES = ["psms_all.txt", "e2g_QUAL.tsv", "e2g_QUANT.tsv"]

    def run(
        self, file=None, path: Path = None, depth=5, **kws
    ) -> Collection[RunContainer]:

        RESULT_FILES = self.RESULT_FILES
        # res = li()
        logger.debug(f"file:{file} path:{path}")
        res = defaultdict(RunContainer)
        observed_files = list()
        # import ipdb; ipdb.set_trace()
        if file is not None:
            for f in file:
                basename = f.stem
                res[basename].add_file(f.resolve())

        if path is None:
            return res.values()
        logger.debug([x for x in path.glob("*")])

        # IDEA turn into a factory?
        # return Container(**).construct()

        for pat in self.PATTERNS:
            for i in range(depth):
                globstr = "*/" * i + pat
                # bug when depth == 1 and file has moved into its new home directory
                for f in path.glob(globstr):  # TODO fix if path is None
                    # if (not f.is_file()) and (not f.is_symlink()):

                    # if f.name.startswith("46119"):
                    #    import ipdb

                    #    ipdb.set_trace()

                    logger.debug(f"pat:{pat}, file:{f}")
                    if f.is_dir():
                        continue
                    if f.name in observed_files:
                        continue

                    # print(f)
                    # recno, runno, searchno = parse_rawname(f.stem)
                    # if searchno is None:
                    #    searchno = "6"
                    # name=parse_rawname(f.name)
                    # full_name =  f"{recno}_{runno}_{searchno}"

                    basename = f.stem
                    logger.debug(f"0) {basename}")
                    for ext in self.FILE_EXTENSIONS:
                        if basename.endswith(ext):
                            basename = basename.split(ext)[0]
                            # break
                    logger.debug(f"1) {basename}")

                    # else:
                    if f.suffix == ".tsv":
                        basename = f.stem
                    # find .tsv files
                    # adds rec_run_search where rec_run match
                    # if any(str(f).endswith(x) for x in RESULT_FILES):
                    #     for k in res:
                    #         # TODO fix
                    #         if k.startswith(basename[:7]):
                    #             res[k].add_file(f)

                    if f.is_symlink():
                        _name = f.absolute()
                    else:
                        _name = f.resolve()

                    if any(x not in f.name for x in RESULT_FILES):
                        res[basename].add_file(_name)
                        # add files to RunContainers with a matching `_name`
                    observed_files.append(f.name)
                    # weird behavior fix later

                    # run_container = RunContainer(stem=f.stem)
                    # res.append(run_container)
        # we really only need the RunContainers,
        # the defaultdict collection made it convienent to keep adding
        # files to the same "base" file

        ret = [x for x in res.values() if x.n_files > 0]
        return ret
