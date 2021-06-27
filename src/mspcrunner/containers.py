from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


class RunContainer:

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
                if all(
                    self._files[x].name[ix] == char for x in range(len(self._files))
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
        return self._stem

    @property
    def rootdir(self):
        if self._rootdir is None:
            parents = {x.parent for x in self._files}
            if len(parents) > 1:

                raise ValueError("cannot handle, but easily fixed")
            self._rootdir = list(parents)[0]
        return self._rootdir

    def add_file(self, f):
        # keep a record of all files

        # always store raw files in "raw"
        print(f)
        if f.name.endswith("raw"):
            self._file_mappings["raw"] = f

        if f.name.endswith("mzML"):
            self._file_mappings["spectra"] = f
        elif f.name.endswith("raw") and self._file_mappings.get("spectra") is None:
            self._file_mappings["spectra"] = f
        elif f.name.endswith("pin"):
            self._file_mappings["pinfile"] = f
        elif f.name.endswith("tsv"):
            self._file_mappings["tsv_searchres"] = f
        elif f.name.endswith("pepXML"):
            self._file_mappings["pepxml"] = f
        elif f.name.endswith("mokapot.psms.txt"):
            self._file_mappings["mokapot-psms"] = f
        elif f.name.endswith("mokapot.peptides.txt"):
            self._file_mappings["mokapot-peptides"] = f
        elif f.name.endswith("SICstats.txt"):
            self._file_mappings["SICs"] = f
        elif f.name.endswith("ReporterIons.txt"):
            self._file_mappings["ReporterIons"] = f
        elif "MSPCRunner" in f.name:
            self._file_mappings["MSPCRunner"] = f
        # elif f.name.endswith('MSPCRunner'):
        # self._file_mappings['ReporterIons'] = f
        else:
            return
            # logger.info(f"Unknown file {f}")
        self._files.append(f)
        self.reset_properties()

    def update_files(self) -> None:
        """"""
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

        for filetype, fileref in self._file_mappings.items():
            file = self.get_file(filetype)

            if not isinstance(file, Path):
                continue

            new_file = new_dir / file.parts[-1]

            print("===*** ", filetype, file.resolve(), new_file.resolve())

            if not file.resolve() == new_file.resolve():
                logger.info(f"{file} -> {new_file}")
                relocated_obj = file.rename(new_file)
                self._file_mappings[filetype] = relocated_obj

            if filetype in ("raw", "spectra"):
                if (
                    self._file_mappings["raw"].suffix
                    == self._file_mappings["spectra"].suffix
                ):
                    self._file_mappings["raw"] = self._file_mappings[
                        "spectra"
                    ] = new_file

        # if filetype in ("raw", "spectra"):
        #     self._file_mappings['raw'] = new_file

        # file = self.get_file(filetype)
