from ipdb import set_trace
import os
from pathlib import Path
import pandas as pd
from warnings import warn
from fnmatch import fnmatch


import ipdb

from .logger import get_logger

logger = get_logger(__name__)

from dataclasses import dataclass
import typing
from typing import List, Collection, Dict

from abc import ABC, ABCMeta, abstractmethod

import re

RECRUN_REGEX = re.compile(r"(\d{5})_(\d+)_(\d+)?")


def parse_rawname(s):
    if isinstance(s, Path):
        s = s.name
    res = RECRUN_REGEX.match(s)

    # logger.debug(f"regex {RECRUN} on {name} returns {res}")
    recno, runno, searchno = None, None, None
    if res is None:  # better error checking here
        return None, None, None
    recno = res.group(1)
    runno = res.group(2)
    searchno = res.group(3)
    return recno, runno, searchno


class AbstractContainer(ABC):
    """Abstract Container designed to collects certain files
    and produce a hash value equal to the files that have been collected"""

    FILE_EXTENSIONS = None
    MAPPING = dict()
    PATTERNS = tuple()
    NEG_PATTERNS = tuple()
    # use these to trim

    def __init__(self) -> None:
        super().__init__()
        self._file_mappings = dict()
        self._files_added: List[Path] = list()
        self._rootdir = None

    @abstractmethod
    def add_file(self, input, **kwargs):
        """
        abstract method for populating a container with files
        """

    def update_rootdir(self):
        parents = {x.parent for x in self._file_mappings.values()}

        if not parents:
            return
        if len(parents) > 1:
            # take the "last" parent, final home?
            # FUTURE what if custom dir is set?
            logger.debug(f"found {len(parents)} parents, using deepest")
            parents = sorted(parents, reverse=True)
            # probably not most ideal way of picking parent

        logger.debug(f"rootdir: {self._rootdir}")

        self._rootdir = list(parents)[0]

        logger.debug(f"rootdir: {self._rootdir}")

        return self._rootdir

    @classmethod
    def make_basename(file):
        pass

    @property
    def n_files(self):
        "number of files added to container"
        return len(self._file_mappings)

    def __hash__(self) -> int:
        # return super().__hash__()
        s = [f"{k}:{v}" for k, v in self._file_mappings.items()]
        val = hash(tuple([hash(x) for x in sorted(s)]))
        return val

    def __eq__(self, other) -> bool:
        "Or just compare hash values?"
        if not isinstance(other, self.__class__):
            return False
        if len(self._file_mappings) != len(other._file_mappings):
            return False
        if set(self._file_mappings.keys()) != set(other._file_mappings.keys()):
            return False
        for k, v in self._file_mappings.items():
            # right now just check if files are same
            if other._file_mappings[k].name != v.name:
                return False
        return True

    @property
    def rootdir(self):
        if self._rootdir is None:
            self.update_rootdir()
        return self._rootdir


class AbstractBaseFactory:
    ...


class RunContainer(AbstractContainer):
    """
    Container for 1 ms run (raw/mzml/perc/ionquant/reporterionquant/...)

    """

    # these are the names to be used for `get_file` to get their corresponding attributes
    MAPPING = dict(
        spectra="_spectra",
        pinfile="_mokapot_psms",
        reporterions="_reporterions",
        # TODO expand
    )
    PATTERNS = ("*raw", "*mzML", "*txt", "*pin", "*tsv")
    NEG_PATTERNS = ("*species_ratios.tsv",)
    # use these to trim
    FILE_EXTENSIONS = [
        ".mokapot.psms",
        ".mokapot.peptides",
        "_ReporterIons",
        "_SICstats",
        "_ScanStats",
        "_ScanStatsConstant",
        "_MSPCRunner_a1",
    ]

    def __init__(self, stem=None, rootdir: Path = None) -> None:
        """
        can set the stem explicitly or let it self-calculate
        :see self.stem:
        :n_files:
        :stem:
        :rootdir:

        :update_rootdir:

        """
        self._stem = stem
        self._files = list()
        self._file_mappings = dict()
        if rootdir is None:
            rootdir = Path(".")
        self._rootdir = rootdir
        self._files_added = 0
        # self._spectra = None
        # self._pinfile = None
        # self._tsv_searchres = None
        # self._pepxml = None
        # self._mokapot_psms = None
        # self._mokapot_peptides = None
        # self._sics = None
        # self._reporterions = None

    @classmethod
    def make_basename(self, f: Path):
        basename = None
        for ext in self.FILE_EXTENSIONS:
            if re.search(ext, f.stem):
                basename = re.sub(f"{ext}.*", "", f.stem)
        # else:
        # do this more cleanly
        if (
            f.suffix == ".tsv"
            or f.suffix == ".raw"
            or f.suffix == ".mzML"
            or f.suffix == ".pin"
        ):
            basename = f.stem
        # if basename is None:
        return basename

    @property
    def n_files(self):
        return len(self._file_mappings)

    def reset_properties(self):
        self._stem = None
        # print('reste', self._stem)
        # self._rootdir = None

    @property
    def spectra(self):
        return

    def __repr__(self) -> str:
        return f"RunContainer <{os.path.join(self._rootdir.__str__(), self.stem.__str__())}>"

    def __str__(self) -> str:
        return f"RunContainer <{os.path.join(self._rootdir.__str__(), self.stem.__str__())}>"

    # def __hash__(self) -> int:
    #     return super().__hash__()

    @property
    def stem(self):
        """stem

        this property is updated if stem is None and file_mappings has at least one entry
        else it returns current stem value

        Returns:
            [str]: [common component of all file names]
        """
        if self._stem is None and self._file_mappings:
            name = ""
            valid_files = self._file_mappings.values()
            valid_files = list(valid_files)
            # for ix, char in enumerate(self._files[0].stem):
            for ix, char in enumerate(list(valid_files)[0].stem):
                if all(
                    valid_files[x].name[ix] == char for x in range(len(valid_files))
                ):
                    name += char
                else:  # stop once we've found a difference
                    break
            _stem = name.strip("_")

            # stem_length = min(len(x.stem) for x in self._files)
            # _stem = self._files[0].name[0:stem_length]
            # stems = {x.stem for x in self._files}
            # if len(stems) > 1:
            #   raise ValueError('!!')
            self._stem = _stem
            # if _stem.endswith("F1"):
            #     pass
        # elif self._stem is None and not self._files:
        #     self._stem = "None"
        return self._stem

    def update_rootdir(self):
        parents = {x.parent for x in self._file_mappings.values()}

        if not parents:
            return
        if len(parents) > 1:
            # take the "last" parent, final home?
            # FUTURE what if custom dir is set?
            logger.debug(f"found {len(parents)} parents, using deepest")
            parents = sorted(parents, reverse=True)
            # probably not most ideal way of picking parent

        logger.debug(f"rootdir: {self._rootdir}")

        self._rootdir = list(parents)[0]

        logger.debug(f"rootdir: {self._rootdir}")

        return self._rootdir

    @property
    def rootdir(self):
        if self._rootdir is None:
            self.update_rootdir()
        return self._rootdir

    def add_file(self, f):
        """
        collects psms related files including:
         - raw
         - mzML
         - pin
         - tsv
           - MSFragger search result exported as tsv
           - MSPCRunner combined output psms file
         - pemXML
         - txt
           - SICstats.txt
           - ScanStats.txt
           - ReporterIons.txt

        """

        if isinstance(f, str):
            f = Path(f)
        # keep a record of all files

        NEG_PATTERNS = self.NEG_PATTERNS
        if any(f.match(pattern) for pattern in NEG_PATTERNS):
            logger.debug(f"skipping {f}")

        # always store raw files in "raw"
        # print(f)
        if f.name.endswith("raw"):
            self._file_mappings["raw"] = f

        if f.name.endswith("mzML"):
            self._file_mappings["spectra"] = f
        # elif f.name.endswith("psms_all.txt"):
        #     self._file_mappings["for-gpg"] = f
        elif f.name.endswith("raw") and self._file_mappings.get("spectra") is None:
            self._file_mappings["spectra"] = f
        elif f.name.endswith("pin"):
            self._file_mappings["pinfile"] = f
        elif f.name.endswith("tsv") and not any(x in f.name for x in ("psm", "e2g")):
            self._file_mappings["tsv_searchres"] = f
        elif f.name.endswith("pepXML"):
            self._file_mappings["pepxml"] = f
        elif f.name.endswith("mokapot.psms.txt"):
            self._file_mappings["mokapot-psms"] = f
        elif f.name.endswith("mokapot.peptides.txt"):
            self._file_mappings["mokapot-peptides"] = f
        elif f.name.endswith("SICstats.txt"):
            self._file_mappings["SICs"] = f
        elif f.name.endswith("ScanStats.txt"):
            self._file_mappings["ScanStats"] = f
        elif f.name.endswith("ReporterIons.txt"):
            self._file_mappings["ReporterIons"] = f
        elif "MSPCRunner" in f.name:
            self._file_mappings["MSPCRunner"] = f
        else:
            pass

        # # gpgroup
        # elif "e2g_QUAL" in f.name:
        #     self._file_mappings["e2g_QUAL"] = f
        # elif "e2g_QUANT" in f.name:
        #     self._file_mappings["e2g_QUANT"] = f

        # elif f.name.endswith('MSPCRunner'):
        # self._file_mappings['ReporterIons'] = f
        # logger.info(f"Unknown file {f}")
        self._files.append(f)
        self.reset_properties()  # forces stem to be recalculated
        self.update_rootdir()

        return self

    # def update_files(self) -> None:
    #     """"""
    #     if self.rootdir is None:
    #         return  # nothing to do
    #     for f in self.rootdir.glob(f"{self.stem}*"):
    #         if f in self._files:
    #             pass
    #         import ipdb

    #         if f.stem.endswith("F1") and "F1" in f.name:
    #             ipdb.set_trace()
    #         self.add_file(f)

    def update_files(self):
        pass

    def get_file(self, name: str):
        # can expand this to different methods for getting different files, with various checks
        # Can add more logic such as checking if file exists, file size, creation time, etc
        # self.update_rootdir()

        # if name in ("raw", "spectra"):
        #     if self._file_mappings['raw'].name() == self._file_mappings['spectra'].name():
        #         self._file_mappings['raw'] = self._file_mappings['spectra'].resolve()
        file = self._file_mappings.get(name, None)
        if file is None:
            return None

        return file

        # return self.attrs.get(name, lambda x: x)()

    def relocate(self, new_dir: Path):
        logger.info(f"relocating {self} to {new_dir}")

        # self.update_files()

        for filetype, fileref in self._file_mappings.items():
            # file = self.get_file(filetype)

            if not isinstance(fileref, Path):
                continue

            if new_dir == fileref.parent:
                continue  # already in the right place
            new_file = new_dir / fileref.parts[-1]

            if new_file.exists():
                logger.warning(f"{new_file} already exists, overwriting")
            logger.info(f"{fileref} -> {new_file}")

            #
            try:
                relocated_obj = fileref.rename(new_file)
            except Exception as e:
                raise e

            self._file_mappings[filetype] = relocated_obj

            if filetype in ("raw", "spectra"):
                if (
                    self._file_mappings.get("raw")
                    and self._file_mappings["raw"].suffix
                    == self._file_mappings["spectra"].suffix
                ):
                    self._file_mappings["raw"] = new_file
                    self._file_mappings["spectra"] = new_file

        # self.update_files()
        # if filetype in ("raw", "spectra"):
        #     self._file_mappings['raw'] = new_file

        # file = self.get_file(filetype)

    def load(self):
        pass


class SampleRunContainer(AbstractContainer):
    """
    Container for final mass spec psms, proteins, gpgroups,
    etc, results

    """

    # def __init__(self, stem=None, rootdir: Path = None) -> None:
    #     """
    #     can set the stem explicitly or let it self-calculate
    #     :see self.stem:
    #     """
    #     self._stem = stem
    #     self._files = list()
    #     self._file_mappings = dict()
    #     self._rootdir = rootdir
    #     self._files_added = 0
    #     self.runcontainers = None
    FILE_EXTENSIONS = ["tsv"]
    PATTERNS = ["*tsv", "*txt", "*tab", "*gct", "*xlsx", "*xls"]

    def __init__(
        self,
        rootdir=Path("."),
        record_no=None,
        run_no=1,
        search_no=6,
        stem=None,
        runcontainers=None,
        **kwargs,
    ) -> None:
        self.name = ""
        if "name" in kwargs:
            self.name = kwargs.pop("name")
        super().__init__(**kwargs)

        self.runcontainers = runcontainers or tuple()
        self.stem = None
        # self.runcontainers: Collection[RunContainer] = attr.ib(
        #     default=(RunContainer(),)
        # )
        self._file_mappings: Dict[str, Path] = dict()
        self.psms_file = None

        # self.rootdir = attr.ib(default=Path("."))
        # self.record_no = record_no
        if rootdir is None:
            rootdir = Path(".")
        self._rootdir = rootdir

        self._record_no = record_no
        self._run_no: int = run_no
        self._search_no: int = search_no

        self.phospho: bool = False
        self.labeltype: str = "none"
        self.refseq: Path = None

        self._psms_file: Path = None

    def set_runno(self):
        ...

    def reset_mspc_files(self):
        self.runcontainers = None

    def __str__(self):
        return f"SampleRunContainer: {self.record_no}_{self.run_no}_{self.search_no}"

    def __repr__(self):
        return f"SampleRunContainer: {self.record_no}_{self.run_no}_{self.search_no}"

    @property
    def rec_run_search(self):
        return f"{self.record_no}_{self.run_no}_{self.search_no}"

    @property
    def record_no(self):
        self._update_recrunsearch()
        if self._record_no is not None:
            return self._record_no

        # filemappings = {parse_rawname(x) for x in self._file_mappings.values()}
        # assert len(filemappings) < 2
        ## filemappings is a list of tuples (recno, runno, searchno)
        # if filemappings:
        #    self._record_no = list(filemappings)[0][0]
        # return self._record_no

    @property
    def run_no(self):
        self._update_recrunsearch()
        if self._run_no is not None:
            return self._run_no

    @property
    def search_no(self):
        self._update_recrunsearch()
        if self._search_no is not None:
            return self._search_no

    def _update_recrunsearch(self):
        filemappings = {parse_rawname(x) for x in self._file_mappings.values()}
        if len(filemappings) == 0:
            return

        assert len(filemappings) == 1  # we want 1 and only 1 result
        self._record_no = list(filemappings)[0][0]
        self._run_no = list(filemappings)[0][1]
        self._search_no = list(filemappings)[0][2]

    @property
    def mspcfiles(self):
        _mspcfiles = [
            container.get_file("MSPCRunner") for container in self.runcontainers
        ]
        _mspcfiles = filter(None, _mspcfiles)
        return _mspcfiles

    def check_psms_files(self) -> None:
        """check_psms_files check and log MSPCRunner files in run containers

        currently only logs all files

        could be a good place to extend checks

        """
        files = set(
            filter(None, [x.get_file("MSPCRunner") for x in self.runcontainers])
        )
        # import ipdb; ipdb.set_trace()

        allfiles = [x.get_file("MSPCRunner") for x in self.runcontainers]

        logger.info(f"{self}: nfiles : {len(files)}")

        for f in sorted(files):
            logger.info("\b" * 20 + "\t" + str(f))
            # f"{"\b"*20}\t{f}")

    @property
    def psms_filePath(self) -> Path:
        self.update_rootdir()
        psms_file = self.get_file("psms_all")

        if psms_file is None:
            outname = Path(
                f"{self.record_no}_{self.run_no}_{self.search_no}_psms_all.txt"
            )
        elif isinstance(psms_file, Path):
            outname = psms_file.name
        else:
            # ?
            outname = Path(outname)

        outpath = self.rootdir / Path(outname)  # this works if rootdir is set correctly
        # could check if something exists

        return outpath

    def concat(self, force=False):
        if self.mspcfiles is None or len(list(self.mspcfiles)) == 0:
            logger.info(f"nothing to concat")
            return

        outpath = self.psms_filePath
        if outpath.exists() and not force:
            logger.info(f"{outpath} already exists, not writing")
            return

        df = pd.concat(pd.read_table(f).assign(rawfile=f.name) for f in self.mspcfiles)
        df.to_csv(outpath, sep="\t", index=False)
        logger.info(f"Wrote {outpath}")

        self.psms_file = outpath

    def _check_stem(self, attribute, value):
        return

    def add_file(self, f):
        if isinstance(f, str):
            f = Path(f)
        # keep a record of all files
        # always store raw files in "raw"
        # if f.name.endswith("raw"):
        #     self._file_mappings["raw"] = f
        keywords = {
            "psms_all": "*_psms_all*txt",
            "e2g_QUAL": "*e2g_QUAL*tsv",
            "e2g_QUANT": "*e2g_QUANT*tsv",
            "psms_QUAL": "*psm_QUAL*tsv",
            "psms_QUANT": "*psms_QUANT*tsv",
            "site_table_nr": "*site_table_nr*gct",
        }  # we strip off the pattern matches after
        for kw_name, thekw in keywords.items():
            # kw_name = thekw.strip("tsv").strip("gct").strip("*")
            if fnmatch(f.name, thekw):
                if kw_name in self._file_mappings.keys():
                    old_kw = self._file_mappings[kw_name]
                    logger.warning(self)
                    logger.warning(
                        f"when setting {kw_name}, overwriting {old_kw} with {f}"
                    )
                logger.info(f"setting {kw_name} to {f}")
                self._file_mappings[kw_name] = f
        else:
            logger.debug(f"not setting {f} to any file")
        return self
        # if "psms_all" in f.name:
        #    self._file_mappings["input_psms"] = f
        # elif "e2g_QUAL" in f.name:
        #    self._file_mappings["e2g_QUAL"] = f
        # elif "e2g_QUANT" in f.name:
        #    self._file_mappings["e2g_QUANT"] = f
        # else:
        #     pass

        self.reset_properties()  # forces stem to be recalculated
        self.update_rootdir()

    def get_file(self, name):
        return self._file_mappings.get(name)

    @classmethod
    def make_basename(self, file) -> str:
        res = parse_rawname(file.name)

        res = [*filter(None, res)]
        if len(res) == 3:
            return f"{res[0]}_{res[1]}_{res[2]}"

        if len(res) == 2:  # or
            return f"{res[0]}_{res[1]}"
        else:
            return None

    # would be better to inherit?
    def __hash__(self) -> int:
        # return super().__hash__()
        s = [f"{k}:{v}" for k, v in self._file_mappings.items()]
        # let's also include runcontainers
        for rc in self.runcontainers:
            s.append(hash(rc))
        val = hash(tuple([hash(x) for x in sorted(s)]))
        return val

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if len(self._file_mappings) != len(other._file_mappings):
            return False
        if set(self._file_mappings.keys()) != set(other._file_mappings.keys()):
            return False
        for k, v in self._file_mappings.items():
            # right now just check if files are same
            if other._file_mappings[k].name != v.name:
                return False
        return True
