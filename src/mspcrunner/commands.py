# commands
import sys
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
import logging
import re
from time import time

# from collections.abc import Iterable
from collections import OrderedDict
from typing import Tuple, Any, Iterable, List, Dict

BASEDIR = os.path.split(__file__)[0]

MSFRAGGER_EXE = os.path.abspath(
    os.path.join(BASEDIR, "../../ext/MSFragger/MSFragger-3.2/MSFragger-3.2.jar")
)
MSFRAGGER_EXE = Path(MSFRAGGER_EXE)

# MSFRAGGER_EXE = os.path.join(BASEDIR, "..\ext\MSFragger\MSFragger-3.2\MSFragger-3.2.jar")
MASIC_EXE = os.path.abspath(os.path.join(BASEDIR, "../../ext/MASIC/MASIC_Console.exe"))
MASIC_EXE = Path(MASIC_EXE)

MOKAPOT = "mokapot"


def get_logger(name=__name__):

    # import queue
    # from logging import handlers
    # que = queue.Queue(-1)  # no limit on size
    # queue_handler = handlers.QueueHandler(que)
    # handler = logging.StreamHandler()
    # listener = handlers.QueueListener(que, handler)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # logger.addHandler(queue_handler)

    fh = logging.FileHandler("MSPCRunner.log")
    # fh.flush = sys.stdout.flush
    # fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    # ch.flush = sys.stdout.flush
    # ch = logging.StreamHandler()

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # listener.start()
    # logging.getLogger('').addHandler(fh)
    return logger


logger = get_logger()

# from pypattern import Command, Invoker, Receiver


from abc import ABCMeta, abstractmethod


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
        logger.info(
            f"{self!r} is pretending to run {CMD}\n\targs:{args}, kwargs:{kwargs}"
        )
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
        self.inputfiles = inputfiles
        self.paramfile = paramfile
        self.outdir = outdir or Path(".")

    def __repr__(self):
        return f"{self.NAME} | {self.name}"

    def announce(self) -> None:
        logger.info(f"Setting up {self.NAME}")

    def update_inputfiles(self, inputfiles) -> None:
        self.inputfiles = inputfiles

    def get_inputfiles(self) -> List[Path]:
        if self.inputfiles is None:
            return (Path("<inputfiles>"),)
        return self.inputfiles

    def execute(self):
        raise NotImplementedError("need to define")

    @property
    def CMD(self):
        return None

    def execute(self):
        "execute"
        # return self._receiver.action("run", self.CMD)
        return self._receiver.run(CMD=self.CMD)


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

    def run(self, inputfiles=None, outdir=None, CMD=None) -> Dict[Path, List[Path]]:
        """
        :inputfiles: Path objects of files to
        """
        results = defaultdict(list)
        for inputfile in inputfiles:
            recno, runno, searchno = parse_rawname(inputfile.name)
            outname = "_".join(filter(None, (recno, runno, searchno)))
            new_home = outdir / outname
            new_home.mkdir(exist_ok=True)
            logger.info(f"{inputfile} -> {new_home}")
            logger.info("For future")
            results[new_home].append(inputfile)
            return results


class PythonCommand(Command):

    NAME = "PythonCommand"

    @property
    def CMD(self):
        return dict(
            inputfiles=self.inputfiles,
            outdir=self.outdir,
        )

    def execute(self):
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

    def find_pinfiles(self):
        """
        looks for pinfiles in same directory as input (raw) files
        """
        pinfiles = list()
        for inputfile in self.inputfiles:  # Path objects
            _pinfile_glob = inputfile.parent.glob("*pin")
            for _pinfile in _pinfile_glob:
                pinfiles.append(_pinfile)
        self.pinfiles = pinfiles
        return pinfiles

    @property
    def CMD(self):
        # self.announce()

        # for inputfile in self.get_inputfiles():
        # for inputfile in self.inputfiles:

        # parse_rawname
        return [
            [
                MOKAPOT,
                "--decoy_prefix",
                "rev_",
                "--missed_cleavages",
                "2",
                "--dest_dir",
                self.outdir,
                "--file_root",
                f"{pinfile.stem}_",
                "--train_fdr",
                self.train_fdr,
                "--test_fdr",
                self.test_fdr,
                "--seed",
                self.seed,
                "--folds",
                self.folds,
                pinfile,
            ]
            for pinfile in self.pinfiles
        ]

    def execute(self):
        "execute"
        # return self._receiver.action("run", self.CMD)
        self.find_pinfiles()  # should be created by the time this executs
        for CMD in self.CMD:
            ret = self._receiver.run(CMD=CMD)


class MASIC(Command):

    NAME = "MASIC"

    @property
    def CMD(self):
        # TODO check os
        self.announce()
        for inputfile in self.inputfiles:
            return [
                "mono",
                MASIC_EXE,
                f"/P:{self.paramfile}",
                f"/O:{inputfile.parent.resolve()}",
                f"/I:{inputfile.resolve()}",
            ]
            # return [MASIC_EXE, f"/P:{self.paramfile}", f"/O:{self.outdir}", f"/I:{inputfile}"]
            # return [MASIC_EXE, f"/P:\"{self.paramfile}\"", f"/O:{self.outdir}", f"/I:{inputfile}"]


class MSFragger(Command):

    NAME = "MSFragger"

    def __init__(self, *args, ramalloc="50G", **kwargs):
        super().__init__(*args, **kwargs)
        self.ramalloc = ramalloc
        self._params = None

    def read_config(self, conf) -> dict: # this is smarter than using csv module
        parser = ConfigParser(inline_comment_prefixes="#")
        with open(conf) as stream:
            parser.read_string("[top]\n" + stream.read())  # create dummy header
        return parser["top"]

    @property
    def params() -> dict:
        if self._params is None:
            param_dict = read_params(self.conf)
            self._params = param_dict
        return self._params

    def get_param_value(self, param):
        return self._params.get(param)

    def set_param(self, param, value):
        if param not in self._params:
            logging.error(f"{param} does not exist")
        self._params[param] = value

    @staticmethod
    def quote_if_windows(x): # not needed
        if platform.system() == "Windows":
            return f'"{x}"'
        return f"{x}"

    @property
    def CMD(self):
        return [
            "java",
            f"-Xmx{self.ramalloc}G",
            "-jar",
            #self.quote_if_windows(MSFRAGGER_EXE),
            #self.quote_if_windows(self.paramfile.resolve()),
            #f"\"{MSFRAGGER_EXE}\"",
            f"{MSFRAGGER_EXE.resolve()}",
            f"{self.paramfile.resolve()}",
            *self.get_inputfiles(),
        ]


class MASIC_Tester(MASIC):

    NAME = "MASIC Tester"

    def execute(self):
        print(f"{self!r} : sending {self.CMD}")
        # ret = self._receiver.action("run", self.CMD)
        ret = self._receiver.run(self.CMD)
        # print(f"{self} : sending {self.CMD}")


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


class Worker:  # invoker
    """
    Invoker
    """

    def __init__(
        self,
    ):  # Override init to initialize an Invoker with Commands to accept.
        self._history = list()
        self._commands = dict()
        self._output = OrderedDict()

    def register(self, command_name, command):
        self._commands[command_name] = command

    def execute(self, command_name):
        "Execute any registered commands"

        if command_name in self._commands.keys():
            output = self._commands[command_name].execute()
            self._history.append((time(), command_name))
            self._output[command_name] = output
            return output

        else:
            logger.error(f"Command [{command_name}] not recognised")

    def something_else(self):
        pass

    def set_on_start(self, command):
        pass
