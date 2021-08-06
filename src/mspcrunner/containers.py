from os import initgroups
from pathlib import Path
import pandas as pd

import ipdb

from .logger import get_logger

logger = get_logger(__name__)

from dataclasses import dataclass
import typing
from typing import List, Collection, Dict
import attr

from abc import ABC, ABCMeta, abstractmethod

import re

RECRUN_REGEX = re.compile(r"(\d{5})_(\d+)_(\d+)")


def parse_rawname(s):
    res = RECRUN_REGEX.match(s)
    # logger.debug(f"regex {RECRUN} on {name} returns {res}")
    recno, runno, searchno = None, None, None
    if res is None:
        return None, None, None
    recno = res.group(1)
    runno = res.group(2)
    searchno = res.group(3)
    return recno, runno, searchno


class AbstractContainer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._file_mappings = dict()
        self._files_added: List[Path] = list()

    @abstractmethod
    def add_file(self, input, **kwargs):
        """
        abstract method for populating a container with files
        """

    @property
    def n_files(self):
        return len(self._file_mappings)

    def __hash__(self) -> int:
        # return super().__hash__()
        s = [f"{k}:{v}" for k, v in self._file_mappings.items()]
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

    def __init__(self, stem=None, rootdir: Path = None) -> None:
        """
        can set the stem explicitly or let it self-calculate
        :see self.stem:
        """
        self._stem = stem
        self._files = list()
        self._file_mappings = dict()
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

    @property
    def n_files(self):
        return len(self._file_mappings)

    def reset_properties(self):
        self._stem = None
        # print('reste', self._stem)
        self._rootdir = None

    @property
    def spectra(self):
        return

    def __repr__(self) -> str:
        return f"RunContainer <{self.stem}>"

    def __str__(self) -> str:
        return f"RunContainer <{self.stem}>"

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

        self._rootdir = list(parents)[0]
        return self._rootdir

    @property
    def rootdir(self):
        if self._rootdir is None:
            self.update_rootdir()
        return self._rootdir

    def add_file(self, f):

        if isinstance(f, str):
            f = Path(f)
        # keep a record of all files

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
        self.reset_properties()

        return self

    def update_files(self) -> None:
        """"""
        if self.rootdir is None:
            return  # nothing to do
        for f in self.rootdir.glob(f"{self.stem}*"):
            if f in self._files:
                pass
            self.add_file(f)

    def get_file(self, name: Path):
        # can expand this to different methods for getting different files, with various checks
        # Can add more logic such as checking if file exists, file size, creation time, etc
        self.update_files()

        # if name in ("raw", "spectra"):
        #     if self._file_mappings['raw'].name() == self._file_mappings['spectra'].name():
        #         self._file_mappings['raw'] = self._file_mappings['spectra'].resolve()
        return self._file_mappings.get(name)

        # return self.attrs.get(name, lambda x: x)()

    def relocate(self, new_dir: Path):

        self.update_files()

        for filetype, fileref in self._file_mappings.items():
            # file = self.get_file(filetype)

            if not isinstance(fileref, Path):
                continue

            new_file = new_dir / fileref.parts[-1]

            # print("===*** ", filetype, file.resolve(), new_file.resolve())
            # print("===*** ", filetype, file.absolute(), new_file.absolute())

            if not fileref.absolute() == new_file.absolute():

                logger.info(f"{fileref} -> {new_file}")
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

        self.update_rootdir()
        self.update_files()
        # if filetype in ("raw", "spectra"):
        #     self._file_mappings['raw'] = new_file

        # file = self.get_file(filetype)

    def load(self):
        pass


# @attr.s()
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

    def __init__(
        self, rootdir=Path("."), record_no=None, stem=None, runcontainers=None, **kws
    ) -> None:
        super().__init__()

        self.runcontainers = runcontainers
        self.name: str = ""
        self.stem = None
        # self.runcontainers: Collection[RunContainer] = attr.ib(
        #     default=(RunContainer(),)
        # )
        self._file_mappings: Dict[str, Path] = dict()
        self.psms_file = None

        # self.rootdir = attr.ib(default=Path("."))
        self.rootdir = rootdir

        self.record_no: int = record_no
        self.run_no: int = "1"
        self.search_no: int = "6"

        self.phospho: bool = False
        self.labeltype: str = "none"
        self.refseq: Path = None

        self._psms_file: Path = None

    # @stem.validator

    def set_recrunsearch(self):

        psms_file = self._file_mappings.get("input_psms")
        self.psms_file = psms_file  # fix this
        rec, run, search = parse_rawname(psms_file.name)
        self.record_no = rec
        self.run_no = run
        self.search_no = search

    def set_runno(self):
        ...

    def reset_mspc_files(self):
        self.runcontainers = None

    def __str__(self):
        return f"SampleRunContainer: {self.name}"

    def __repr__(self):
        return f"SampleRunContainer: {self.name}"

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
        files = [x.get_file("MSPCRunner") for x in self.runcontainers]
        logger.info(f"{self}:")
        for f in sorted(files):
            logger.info("\b" * 20 + "\t" + str(f))
            # f"{"\b"*20}\t{f}")

    @property
    def psms_filePath(self) -> Path:

        if self._psms_file is None:
            outname = f"{self.record_no}_{self.run_no}_{self.search_no}_psms_all.txt"
        elif isinstance(self._psms_file, Path):
            outname = self._psms_file.name
        outpath = self.rootdir / Path(outname)
        return outpath

    def concat(self):

        outpath = self.psms_filePath
        if outpath.exists():
            logger.info(f"{outpath} already exists, not writing")
            return

        df = pd.concat(pd.read_table(f) for f in self.mspcfiles)
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
        # print(f)
        # if f.name.endswith("raw"):
        #     self._file_mappings["raw"] = f
        if "psms_all" in f.name:
            self._file_mappings["input_psms"] = f

        elif "e2g_QUAL" in f.name:
            self._file_mappings["e2g_QUAL"] = f
        elif "e2g_QUANT" in f.name:
            self._file_mappings["e2g_QUANT"] = f

        return self

    # would be better to inherit?
    def __hash__(self) -> int:
        # return super().__hash__()
        s = [f"{k}:{v}" for k, v in self._file_mappings.items()]
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
