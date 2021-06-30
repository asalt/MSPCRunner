from collections import defaultdict
from typing import List, Collection

from .containers import RunContainer


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

    def run(self, file=None, path=None, depth=5, **kws) -> Collection[RunContainer]:

        # res = li()
        res = defaultdict(RunContainer)
        observed_files = list()
        # import ipdb; ipdb.set_trace()
        if file is not None:
            for f in file:
                basename = f.stem
                res[basename].add_file(f.resolve())

        if path is None:
            return res.values()

        for pat in self.PATTERNS:
            for i in range(depth):
                globstr = "*/" * i + pat
                # bug when depth == 1 and file has moved into its new home directory
                for f in path.glob(globstr):  # TODO fix if path is None
                    if not f.is_file():
                        continue
                    # print(f)
                    # recno, runno, searchno = parse_rawname(f.stem)
                    # if searchno is None:
                    #    searchno = "6"
                    # name=parse_rawname(f.name)
                    # full_name =  f"{recno}_{runno}_{searchno}"

                    if f.name in observed_files:
                        continue
                    basename = f.stem
                    for ext in self.FILE_EXTENSIONS:
                        if basename.endswith(ext):
                            basename = basename.split(ext)[0]
                            break
                    # else:
                    if f.suffix == ".tsv":
                        basename = f.stem
                    # find .tsv files

                    res[basename].add_file(f.resolve())
                    observed_files.append(f.name)
                    # run_container = RunContainer(stem=f.stem)
                    # res.append(run_container)
        # we really only need the RunContainers,
        # the defaultdict collection made it convienent to keep adding
        # files to the same "base" file
        return [x for x in res.values()]
