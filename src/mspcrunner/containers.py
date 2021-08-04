from os import initgroups
from pathlib import Path
import pandas as pd

import ipdb

from .logger import get_logger

logger = get_logger(__name__)

from dataclasses import dataclass
import typing
import attr

from abc import ABC, abstractmethod


class AbstractBaseFactory:
    ...


class RunContainer:
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

    @property
    def stem(self):
        if self._stem is None and self._files:

            name = ""
            for ix, char in enumerate(self._files[0].stem):
                file_list = [
                    x for x in self._file_mappings.keys() if isinstance(x, Path)
                ]
                if all(file_list[x].name[ix] == char for x in range(len(file_list))):
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
        # keep a record of all files

        # always store raw files in "raw"
        # print(f)
        if f.name.endswith("raw"):
            self._file_mappings["raw"] = f

        if f.name.endswith("mzML"):
            self._file_mappings["spectra"] = f
        elif f.name.endswith("psms_all.txt"):
            self._file_mappings["for-gpg"] = f
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

        # gpgroup
        elif "e2g_QUAL" in f.name:
            self._file_mappings["e2g_QUAL"] = f
        elif "e2g_QUANT" in f.name:
            self._file_mappings["e2g_QUANT"] = f

        # elif f.name.endswith('MSPCRunner'):
        # self._file_mappings['ReporterIons'] = f
        else:
            pass
            # logger.info(f"Unknown file {f}")
        self._files.append(f)
        self.reset_properties()

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
                except:
                    import ipdb

                    ipdb.set_trace()
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


@attr.s()
class SampleRunContainer:
    """
    Container for final mass spec psms, proteins, gpgroups,
    etc, results

    """

    name: str = attr.ib(default=None)
    stem = attr.ib(default=None)
    runcontainers = attr.ib(default=(RunContainer(),))
    psms_file = attr.ib(default=None)

    rootdir = attr.ib(default=Path("."))

    record_no: int = attr.ib(default=None)
    run_no: int = attr.ib(default="1")
    search_no: int = attr.ib(default="6")

    phospho: bool = attr.ib(default=False)
    labeltype: str = attr.ib(default="none")
    refseq: Path = None

    _psms_file: Path = None

    # @stem.validator

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
