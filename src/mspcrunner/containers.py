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

    def __init__(self, stem=None) -> None:
        """
        can set the stem explicitly or let it self-calculate
        :see self.stem:
        """
        self._stem = stem
        self._files = list()
        self._file_mappings = dict()
        # self._spectra = None
        # self._pinfile = None
        # self._tsv_searchres = None
        # self._pepxml = None
        # self._mokapot_psms = None
        # self._mokapot_peptides = None
        # self._sics = None
        # self._reporterions = None

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
            stem_length = min(len(x.stem) for x in self._files)
            _stem = self._files[0].name[0:stem_length]
            stems = {x.stem for x in self._files}
            # if len(stems) > 1:
            #    raise ValueError('!!')
            # self._stem = tuple(stems)[0]
            self._stem = _stem
        return self._stem

    def add_file(self, f):
        # keep a record of all files
        self._files.append(f)

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
        # elif f.name.endswith('MSPCRunner'):
        # self._file_mappings['ReporterIons'] = f
        else:
            pass
            # logger.info(f"Unknown file {f}")

    def get_file(self, name):
        # can expand this to different methods for getting different files, with various checks
        # Can add more logic such as checking if file exists, file size, creation time, etc

        return self._file_mappings.get(name)

        # return self.attrs.get(name, lambda x: x)()

    def relocate(self, new_dir: Path):
        # doesn't actually relocate. this is in FileMover
        # should probably combine here

        for filetype, file in self._file_mappings.items():
            if not isinstance(file, Path):
                continue

            new_file = new_dir / file.parts[-1]
            if not file == new_file:
                logger.info(f"{file} -> {new_file}")
                relocated_obj = file.rename(new_file)
                self._file_mappings[filetype] = relocated_obj
            # file = self.get_file(filetype)
