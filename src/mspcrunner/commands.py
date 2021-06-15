# commands
import logging
import os
import platform
import re
import subprocess
import sys

# from collections.abc import Iterable
from collections import OrderedDict, defaultdict
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from .worker import Worker

import click

BASEDIR = os.path.split(__file__)[0]

MSFRAGGER_EXE = os.path.abspath(
    os.path.join(BASEDIR, "../../ext/MSFragger/MSFragger-3.2/MSFragger-3.2.jar")
)
MSFRAGGER_EXE = Path(MSFRAGGER_EXE)


MOKAPOT = "mokapot"


from .logger import get_logger

logger = get_logger(__name__)


def parse_rawname(name: str) -> Tuple[str, str, str]:
    """yield up to the first 3 numbers in a string separated by underscore
    returns None when number is missing / interrupted
    """
    namesplit = name.split("_")
    yield_counter = 0
    for x in namesplit:
        if x.isnumeric() and yield_counter < 3:
            yield_counter += 1
            yield x
        else:
            break
    while yield_counter < 3:
        yield_counter += 1
        yield None


def run_and_log(CMD, *args, **kwargs):
    """
    log CMD
    run CMD with subprocess.run
    log exitcode
    *args, **kwargs - passed to subprocess.run"""

    logger.info(f"Running: {' '.join(map(str, CMD))}")
    if "capture_output" not in kwargs:
        kwargs["capture_output"] = True
    ret = subprocess.run(CMD, *args, **kwargs)
    retcode = ret.returncode
    logger.info(f"ExitCode: {retcode}")
    if retcode != 0:
        logger.error(f"{retcode}")
        stderr = ret.stderr.decode("UTF8")
        logger.error(f"stderr: {stderr}")

    return ret


def resolve_if_path(x):
    if isinstance(x, Path):
        return x.resolve()
    return x


class CMDRunner:  # receiver
    """
    receiver for running CMD via subprocess
    """

    NAME = "CMDRunner"

    def run(self, *args, CMD: Iterable[Any] = None, **kwargs) -> subprocess.Popen:
        """
        CMD : command to run via subprocess
        *args, **kwargs: passed to subprocess.run
        """
        if CMD is None:
            raise ValueError("Must pass an iterable command collection")

        # logger = logging.getLogger('commands.CMDrunner.run')
        # logger.info(f"Running: {' '.join(CMD)}")

        _CMD = map(resolve_if_path, CMD)
        # _CMD = " ".join(map(str, _CMD))
        _CMD = map(str, _CMD)
        _CMD = list(_CMD)

        logger.info(f"Running: {_CMD}")

        if "capture_output" not in kwargs:
            kwargs["capture_output"] = True

        popen = subprocess.Popen(
            _CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            shell=False,
        )
        # print('made popen', flush=True)

        # while True:
        #    line = popen.stdout.readline().rstrip()
        #    line_e = popen.stderr.readline().rstrip()  # this works
        #    logger.info(line)
        #    logger.warning(line_e)
        #    if not line:
        #        break
        # popen.stdout.close()

        # stdout = list()
        # for stdout_line in popen.stdout:
        #    print(stdout_line, end='')
        # stdout.append(stdout_line)
        # for stdout_line in iter(popen.stdout.readline, ''):
        # while popen.poll() is None:
        stdout = list()
        for stdout_line in popen.stdout:
            logger.info(stdout_line.strip())
            # stdout.append(stdout_line)
            # logger.handlers[0].flush()
            # logger.handlers[1].flush()
            # print(stdout_line, flush=True)
        ## fix this later?

        # for i in stdout:
        # logger.info(i)
        retcode = popen.wait()
        # popen.stdout.close()
        # logging.info(i)

        # run(CMD, *args, **kwargs)

        # retcode = popen.wait()
        if retcode == 0:
            logger.info(f"Command finished with exitcode: {retcode}")
        else:
            logger.error(f"Command finished with exitcode: {retcode}")
        # if retcode != 0:
        # logger.error(f"{retcode}")
        # logger.error(f"{stdout[-1]}")
        # if stderr:  # do we get this here or is it all going to be in the above
        #    # logger.info call?
        #    stderr = popen.stderr.decode("UTF8")
        #    logger.error(f"stderr: {stderr}")

        return popen


class MokaPotRunner:  # receiver
    pass


#    def run(self, pinfiles):
#        for pinfile in pinfiles:
#            psms = mokapot.read_pin(pinfile)
#            results, models = mokapot.brew(psms)
#            results.to_txt()


class CMDRunner_Tester:  # receiver
    """
    receiver that simply logs CMD, *args, **kwargs without actually running anything
    """

    NAME = "Tester"

    def __init__(self, production_receiver=None):
        self.production_receiver = production_receiver
        if production_receiver:
            self.NAME = f"{production_receiver.NAME} Tester"

    def run(self, *args, CMD=None, **kwargs):

        # logger.info(f"Running: {' '.join(map(str, CMD))}")
        # logger.info(
        #    f"{self!r} is pretending to run {CMD}\n\targs:{args}, kwargs:{kwargs}"
        # )
        if "capture_output" not in kwargs:
            kwargs["capture_output"] = True

        return 0


class RawObj:  # command?
    def __init__(self, file):
        # self.basedir = basedir
        if isinstance(file, str):
            file = Path(file)
        self.file = file

    def parse_rawname(self):
        res = self.RECRUN.match(self.file.parts[-1])
        # logger.debug(f"regex {RECRUN} on {name} returns {res}")
        recno, runno = None, None
        if res is None:
            return None, None
        recno = res.group(1)
        runno = res.group(2)
        return recno, runno

    def fetch_files(self, path):
        files = get_folderstats(path)

        files["recno"] = files.recno.astype(float)
        files["runno"] = files.runno.astype(float)


class Command:

    NAME = "Command"

    def __init__(
        self,
        receiver,
        inputfiles=None,
        paramfile=None,
        outdir=None,
        name=None,
        **kwargs,
    ):
        self.name = name
        self._receiver = receiver
        self._CMD = None
        self.inputfiles = inputfiles
        self.paramfile = paramfile
        self.outdir = outdir or Path(".")
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_files(self, inputfiles: dict):
        logger.info(f"Updating inputfiles on {self}")
        self.inputfiles = inputfiles

    def __repr__(self):
        return f"{self.NAME} | {self.name}"

    def announce(self) -> None:
        logger.info(f"Setting up {self.NAME}")

    # def update_inputfiles(self, inputfiles) -> None:
    #     self.inputfiles = inputfiles

    # def get_inputfiles(self) -> List[Path]:
    #     if self.inputfiles is None:
    #         return (Path("<inputfiles>"),)
    #     return self.inputfiles

    # def execute(self):
    #     raise NotImplementedError("need to define")

    @property
    def CMD(self):
        return None

    def execute(self):
        "execute"
        # return self._receiver.action("run", self.CMD)
        return self._receiver.run(CMD=self.CMD)


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
        if self._stem is None:
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

        for filetype, file in self._file_mappings.items():
            if not isinstance(file, Path):
                continue
            relocated_obj = file.rename(new_dir / file.parts[-1])
            logging.info(f"{file} -> {relocated_obj}")
            self._file_mappings[filetype] = relocated_obj
            # file = self.get_file(filetype)


class FileFinder:  # receiver
    NAME = "FileFinder Receiver"

    PATTERNS = ["*raw", "*mzML", "*txt", "*pin"]
    FILE_EXTENSIONS = [".mokapot.psms", "_ReporterIons", "_SICstats", "_MSPCRunner_a1"]

    def run(self, file=None, path=None, depth=5, **kws) -> List[RunContainer]:
        # res = li()
        res = defaultdict(RunContainer)
        for pat in self.PATTERNS:
            for i in range(depth):
                globstr = "*/" * i + pat
                for f in path.glob(globstr):
                    if not f.is_file():
                        continue
                    # print(f)
                    # recno, runno, searchno = parse_rawname(f.stem)
                    # if searchno is None:
                    #    searchno = "6"
                    # name=parse_rawname(f.name)
                    # full_name =  f"{recno}_{runno}_{searchno}"

                    basename = f.stem
                    for ext in self.FILE_EXTENSIONS:
                        if basename.endswith(ext):
                            basename = basename.split(ext)[0]

                    res[basename].add_file((f))
                    # run_container = RunContainer(stem=f.stem)
                    # res.append(run_container)
        return res


class FileMover:  # receiver

    NAME = "FileMover Receiver"
    """execute something internally"""

    def run(self, inputfiles=None, outdir=None, CMD=None, **kws) -> Iterable[Path]:
        # move files command

        logger.info(f"starting {self}")
        if outdir is None:
            outdir = Path(".")
        logger.info(f"Outdir set to {outdir}")

        newfiles = list()
        for inputfile in inputfiles:
            newfile = inputfile.rename(outdir / inputfile.name)
            logger.info(f"{inputfile} --> {newfile}")
            newfiles.append(newfile)

        return newfiles


RECRUN_REGEX = re.compile("(\d{5})_(\d+)_(\d+)")


class FileRealtor:  # receiver
    """
    find a new home for files
    """

    NAME = "FileRealtor"

    def run(
        self, inputfiles: Dict[str, RunContainer] = None, outdir: Path = None, **kwargs
    ) -> Dict[Path, List[Path]]:
        """
        :inputfiles: Path objects of files to
        """
        results = defaultdict(list)
        # print(inputfiles[[x for x in inputfiles.keys()][0]])
        for (
            id,
            run_container,
        ) in inputfiles.items():  # id may be equivalent to stem, but doesn't have to be
            recno, runno, searchno = parse_rawname(run_container.stem)
            if recno is None:
                logger.info(f"Could not find recno for {run_container.stem}, skipping")
                continue
            if searchno is None:
                searchno = "6"
            outname = "_".join(filter(None, (recno, runno, searchno)))

            new_home = outdir / outname
            if not new_home.exists():
                logger.info(f"Creating {new_home}")
                new_home.mkdir(exist_ok=False)
            else:
                # logger.info(f"{new_home} already exists.")
                pass

            # logger.info(f"{inputfile} -> {new_home}")
            run_container.relocate(new_home)

            results[new_home].append(run_container)
        return results


class PythonCommand(Command):

    NAME = "PythonCommand"

    def __init__(self, *args, **kws):
        super().__init__(
            *args, **kws
        )  # this has to be called before we can access "args"

    @property
    def CMD(self):
        """
        Return the dictionary of attributes to pass to the receiver as kwargs.
        """
        self._cmd = self.__dict__
        return self._cmd
        # return dict(
        #    inputfiles=self.inputfiles,
        #    outdir=self.outdir,
        # )

    def execute(self, *args, **kws):
        self.announce()
        return self._receiver.run(**self.CMD)


class MokaPotConsole(Command):

    NAME = "MokaPot Console"

    def __init__(
        self,
        *args,
        decoy_prefix="rev_",
        train_fdr=0.05,
        test_fdr=0.05,
        seed=888,
        folds=3,
        outdir=".",
        **kws,
    ):

        super().__init__(*args, **kws)
        self.decoy_prefix = decoy_prefix
        self.train_fdr = train_fdr
        self.test_fdr = test_fdr
        self.seed = seed
        self.folds = folds
        self.outdir = outdir
        self.pinfiles = tuple()
        self._cmd = None

    def find_pinfiles(self):
        """
        looks for pinfiles in same directory as input (raw) files
        """
        pinfiles = list()
        for inputfile in self.inputfiles:  # Path objects
            _pinfile_glob = inputfile.parent.glob(f"{inputfile.stem}*pin")
            for _pinfile in _pinfile_glob:
                pinfiles.append(_pinfile)
        self.pinfiles = pinfiles
        return pinfiles

    @property
    def CMD(self):
        # self.announce()

        # for inputfile in self.get_inputfiles():
        # for inputfile in self.inputfiles:
        pinfiles = tuple()
        if self.inputfiles:
            pinfiles = [
                x.get_file("pinfile") for x in self.inputfiles.values()
            ]  # the values are RawFile instances
            pinfiles = [x for x in pinfiles if x is not None]

        # parse_rawname
        if not pinfiles:
            res = tuple()
        res = [
            [
                MOKAPOT,
                "--decoy_prefix",
                "rev_",
                "--missed_cleavages",
                "2",
                "--dest_dir",
                self.outdir,
                "--file_root",
                f"{pinfile.stem}",
                "--train_fdr",
                self.train_fdr,
                "--test_fdr",
                self.test_fdr,
                "--seed",
                self.seed,
                "--folds",
                self.folds,
                str(pinfile.resolve()),
            ]
            for pinfile in pinfiles
        ]
        self._CMD = res
        return self._CMD

    def execute(self):
        "execute"
        # return self._receiver.action("run", self.CMD)
        # self.find_pinfiles()  # should be created by the time this executs
        for CMD in self.CMD:
            ret = self._receiver.run(CMD=CMD)


class Percolator(Command):

    # import mokapot

    def execute(self):
        pass
        # self._receiver.run(self.)

    @property
    def CMD(self):
        pass


class Percolator(Command):
    pass


class Triqler(Command):
    pass


class FinalFormatter(Command):
    """
    format results for for gpgrouper running
    """

    pass
