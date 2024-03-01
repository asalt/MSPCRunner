# commands
from ipdb import set_trace
from abc import abstractclassmethod, abstractmethod
import logging
import os
import platform
import re
import subprocess
import sys
from mspcrunner.config import get_conf  # Import the get_conf function


# from collections.abc import Iterable
from collections import OrderedDict, defaultdict
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Collection

import ipdb
from .worker import Worker
from .file_finder import FileFinder
from .containers import RunContainer, SampleRunContainer

import click

import pandas as pd

from RefProtDB.utils import fasta_dict_from_file

from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects import r
from rpy2.robjects import pandas2ri


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
    if name is None:
        name = ""

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


def find_rec_run(target: str):
    "Try to get record, run, and search numbers with regex of a target string with pattern \d+_\d+_\d+"

    _, target = os.path.split(target)  # ensure just searching on filename
    rec_run_search = re.compile(r"^(\d+)_(\d+)_")

    match = rec_run_search.search(target)
    if match:
        recno, runno = match.groups()
        return recno, runno
    return


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


import threading
import queue


def run(fd, q):
    for line in iter(fd.readline, ""):
        q.put(line)
    q.put(None)


def create(fd):
    q = queue.Queue()
    t = threading.Thread(target=run, args=(fd, q))
    t.daemon = True
    t.start()
    return q, t


from subprocess import Popen, PIPE
from threading import Thread
from queue import Queue  # Python 2


def reader(pipe, queue):
    try:
        with pipe:
            for line in iter(pipe.readline, b""):
                queue.put((pipe, line), block=False)
    finally:
        queue.put(None, block=False)


import threading, queue


def merge_pipes(**named_pipes):
    r"""
    lifted from https://stackoverflow.com/a/51668895
    another possible solution:
    https://github.com/waszil/subpiper

    Merges multiple pipes from subprocess.Popen (maybe other sources as well).
    The keyword argument keys will be used in the output to identify the source
    of the line.

    Example:
    p = subprocess.Popen(['some', 'call'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    outputs = {'out': log.info, 'err': log.warn}
    for name, line in merge_pipes(out=p.stdout, err=p.stderr):
        outputs[name](line)

    This will output stdout to the info logger, and stderr to the warning logger
    """

    # Constants. Could also be placed outside of the method. I just put them here
    # so the method is fully self-contained
    PIPE_OPENED = 1
    PIPE_OUTPUT = 2
    PIPE_CLOSED = 3

    # Create a queue where the pipes will be read into
    output = queue.Queue()

    # This method is the run body for the threads that are instatiated below
    # This could be easily rewritten to be outside of the merge_pipes method,
    # but to make it fully self-contained I put it here
    def pipe_reader(name, pipe):
        r"""
        reads a single pipe into the queue
        """
        output.put(
            (
                PIPE_OPENED,
                name,
            )
        )
        try:
            for line in iter(pipe.readline, ""):
                output.put(
                    (
                        PIPE_OUTPUT,
                        name,
                        line.rstrip(),
                    )
                )
        finally:
            output.put(
                (
                    PIPE_CLOSED,
                    name,
                )
            )

    # Start a reader for each pipe
    for name, pipe in named_pipes.items():
        t = threading.Thread(
            target=pipe_reader,
            args=(
                name,
                pipe,
            ),
        )
        t.daemon = True
        t.start()

    # Use a counter to determine how many pipes are left open.
    # If all are closed, we can return
    pipe_count = 0

    # Read the queue in order, blocking if there's no data
    for data in iter(output.get, ""):
        code = data[0]
        if code == PIPE_OPENED:
            pipe_count += 1
        elif code == PIPE_CLOSED:
            pipe_count -= 1
        elif code == PIPE_OUTPUT:
            yield data[1:]
        if pipe_count == 0:
            return


class Receiver:
    pass
    # __slots__ = ("name", "force")

    # def __init__(self, *args, force=False, **kwargs):
    #     self.force = force
    # input_required = None


class CMDRunner(Receiver):  # receiver
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
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            # bufsize=1,
            # universal_newlines=True,
            # shell=False,
            # shell=True,
        )

        # q = Queue()
        # t1 = Thread(target=reader, args=[popen.stderr, q])
        # t2 = Thread(target=reader, args=[popen.stdout, q])
        # t1.daemon = True
        # t2.daemon = True
        # t1.start()
        # t2.start()
        # for _ in range(1):
        #     for source, line in iter(q.get, None):
        #         if not line.strip():
        #             continue
        #         # if not line.strip("."):
        #         #     continue
        #         logger.info(line.strip())
        #         # print("%s: %s" % (source, line))
        # import ipdb

        # for pipe, msg in merge_pipes(stdout=popen.stdout, stderr=popen.stderr):
        #     logger.info(msg)

        retcode = popen.wait()

        # if retcode == 0:
        #     logger.info(f"Command finished with exitcode: {retcode}")
        # else:
        #     logger.error(f"Command finished with exitcode: {retcode}")

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

        # for stderr_line in popen.stderr, stdout_line in popen.stdout:

        # http://amoffat.github.io/sh/sections/asynchronous_execution.html

        # for stdout_line in popen.stdout:
        #     logger.info(stdout_line.strip())
        # for l1, l2 in zip_longest(popen.stdout, popen.stderr):
        #     logger.info(l1.strip())
        #     logger.info(l2.strip())
        # # second guy always runs second, after subprocess completes

        # stdout.append(stdout_line)
        # logger.handlers[0].flush()
        # logger.handlers[1].flush()
        # print(stdout_line, flush=True)
        ## fix this later?

        # for i in stdout:
        # logger.info(i)
        # popen.stdout.close()
        # logging.info(i)

        # run(CMD, *args, **kwargs)

        # retcode = popen.wait()
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
        receiver: Receiver = None,
        paramfile=None,
        outdir=None,
        name=None,
        inputfiles=None,  # deprecitate
        container=None,
        containers=None,  # replace with this
        force=False,
        **kwargs,
    ):
        self.name = name
        self._receiver = receiver
        self.receiver = receiver
        self._CMD = None
        self.inputfiles = inputfiles  # depreciate
        self.containers = containers  # multiple containers to create multiple objects
        self.container = container  # a single container to do something
        self.paramfile = paramfile
        self.force = force
        # self.outdir = outdir or Path(".")
        self.outdir = outdir  # let it stay as none if not given
        for k, v in kwargs.items():
            logger.info(f"Setting {k} to {v}")
            setattr(self, k, v)

    def create(self, runcontainers=None, sampleruncontainers=None, **kwargs):
        if runcontainers is None:
            runcontainers = tuple()
        if sampleruncontainers is None:
            sampleruncontainers = tuple()
        containers = list(runcontainers) + list(sampleruncontainers)

        if containers is None:
            yield self
        else:
            for ix, container in enumerate(containers):
                d = self.__dict__.copy()

                d.update(kwargs)
                d["inputfile"] = container  # a single inputfile
                d["inputfiles"] = container  # depreciate
                d["container"] = container  # depreciate
                d["runcontainers"] = runcontainers
                d["sampleruncontainers"] = sampleruncontainers
                d["name"] = d.get("name", "name") + f"-{ix}"
                if "receiver" not in d and "_receiver" in d:
                    d["receiver"] = d["_receiver"]

                if isinstance(container, SampleRunContainer):
                    pass

                yield type(self)(**d)

    def __repr__(self):
        return f"{self.NAME} | {self.name}"

    def announce(self) -> None:
        # logger.info(f"Setting up {self.NAME}")
        pass

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

    def execute(self, **kwargs):
        "execute"
        # return self._receiver.action("run", self.CMD)

        if not self.CMD:  # CMD lazy loads, can end up empty if all jobs completed
            # does it still do this?
            # Not exactly
            return
        if isinstance(self.CMD, (list, tuple)) and isinstance(
            self.CMD[0], (list, tuple)
        ):
            all_return = list()
            for CMD in self.CMD:
                ret = self._receiver.run(CMD=CMD)
                all_return.append(ret)
        else:
            all_return = self._receiver.run(CMD=self.CMD)
        return all_return


class PythonCommand(Command):
    NAME = "PythonCommand"

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )  # this has to be called before we can access "args"

    def create(self, runcontainers=None, sampleruncontainers=None, **kwargs):
        if runcontainers is None and sampleruncontainers is None:
            yield self
        else:
            if runcontainers is None:
                runcontainers = tuple()
            if sampleruncontainers is None:
                sampleruncontainers = tuple()

            containers = list(runcontainers) + list(sampleruncontainers)
            d = self.__dict__.copy()
            d["containers"] = containers
            d["runcontainers"] = runcontainers
            d["sampleruncontainers"] = sampleruncontainers
            # quick fix

            if "receiver" not in d and "_receiver" in d:
                d["receiver"] = d["_receiver"]
            # ================
            d.update(kwargs)
            if "inputfiles" in d:
                inputfiles = d.pop("inputfiles")
                if inputfiles is not None:
                    pass

            yield PythonCommand(**d)

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

    def execute(self, *args, **kwargs):
        if "force" in kwargs:
            force = kwargs.pop("force")
        cmd = self.CMD
        if hasattr(self, "force"):
            cmd["force"] = self.force
        logger.info(f"Running {self.name} with {self.CMD}")
        return self._receiver.run(**cmd)


class PythonCommandSampleRunContainerFactory(Command):
    NAME = "PythonCommand"

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )  # this has to be called before we can access "args"

    def create(self, runcontainers=None, sampleruncontainers=None, **kwargs):
        receiver = self.receiver
        if "receiver" in kwargs.keys():
            logger.warning(f"overriding receiver with {kwargs['receiver']}")
            receiver = kwargs.pop("receiver")
        if runcontainers is None and sampleruncontainers is None:
            yield self
        for sampleruncontainer in sampleruncontainers:
            newname = f"{self.name}: {sampleruncontainer.rec_run_search}"
            yield PythonCommand(
                sampleruncontainer=sampleruncontainer,
                name=newname,
                receiver=receiver,
                **kwargs,
            )

    # def execute(self, *args, **kws):
    #     return self._receiver.run(**self.CMD)


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
        outdir=None,
        runcontainer=None,
        sampleruncontainer=None,
        runcontainers=None,  # either pass 1 run
        sampleruncontainers=None,  # or 1 sample (collection of runs)
        enzyme="[KR]",
        **kws,
    ):
        super().__init__(*args, **kws)
        self.decoy_prefix = decoy_prefix
        self.train_fdr = train_fdr
        self.test_fdr = test_fdr
        self.seed = seed
        self.folds = folds
        self.outdir = outdir  # not being used at the moment
        self.pinfiles = tuple()
        self._cmd = None
        self.runcontainer = runcontainer
        self.sampleruncontainer = sampleruncontainer
        self.runcontainers = runcontainers  # for construction
        self.sampleruncontainers = sampleruncontainers  # for construction
        self.run_entire_cohort = (
            True  # will preferentially combine runcontainers together
        )
        self.enzyme = enzyme

    # remember this class inherits a create method, which sets self.runcontainers
    def create(self, runcontainers=None, sampleruncontainers=None, **kws):
        if runcontainers is None and sampleruncontainers is None:
            yield self
        else:
            if runcontainers is None:
                runcontainers = tuple()
            if sampleruncontainers is None:
                sampleruncontainers = tuple()

            #
            d = self.__dict__.copy()

            # quick fix
            if "receiver" not in d and "_receiver" in d:
                d["receiver"] = d["_receiver"]

            if self.run_entire_cohort:
                for ix, sampleruncontainer in enumerate(sampleruncontainers):
                    d["sampleruncontainer"] = sampleruncontainer
                    d["name"] = f"mokapot-samplerun-{ix}"
                    yield MokaPotConsole(**d)
            if sampleruncontainers is None:
                for ix, runcontainer in enumerate(runcontainers):
                    d["name"] = f"mokapot-{ix}"
                    d["runcontainer"] = runcontainer
                    yield MokaPotConsole(**d)
            # containers = list(runcontainers) + list(sampleruncontainers)
            # d = self.__dict__.copy()

            # # quick fix
            # if "receiver" not in d and "_receiver" in d:
            #     d["receiver"] = d["_receiver"]
            # # ================
            # for kw in kws:
            #     if kw in d:
            #         d.update(kw, kws[kw])
            # if "inputfiles" in d:
            #     inputfiles = d.pop("inputfiles")
            #     if inputfiles is not None:
            #         pass

            # yield MokaPotConsole(**d, containers=containers)

    @property
    def CMD(self):
        """
        executes on a runcontainer or sampleruncontainer
        """
        if self.runcontainer is None and self.sampleruncontainer is None:
            # need to fail better than this
            return
        # self.announce()

        # for inputfile in self.get_inputfiles():
        # for inputfile in self.inputfiles:
        # self.runcontainer or self.sampleruncontainer should be set
        # output name calculation

        file_root = None
        if self.runcontainer and isinstance(self.runcontainer, RunContainer):
            pinfiles = [self.runcontainer.get_file("pinfile")]
            file_root = pinfiles[0].stem
            outdir = self.runcontainer.rootdir
        elif self.sampleruncontainer and isinstance(
            self.sampleruncontainer, SampleRunContainer
        ):
            pinfiles = [
                runcontainer.get_file("pinfile")
                for runcontainer in self.sampleruncontainer.runcontainers
            ]
            pinfiles = list(filter(None, pinfiles))
            # note this silently excludes runcontainers that don't have a pinfile
            outdir = self.sampleruncontainer.rootdir
        else:
            #!
            # raise ValueError(f"no pinfile found for {self.inputfiles}")
            return
        if len(pinfiles) == 1 and pinfiles[0] is None:
            raise ValueError(f"no pinfile found for {self.inputfiles}")

        # if self.outdir is None and len(pinfiles) == 1:
        if len(pinfiles) == 1:
            file_root = pinfiles[0].stem  # we have to do this iff there is 1 pinfile
        #     outdir = pinfiles[0].parent
        # elif self.outdir is not None and len(pinfiles) > 1:
        #     raise NotImplementedError("Have not resolved multiple file case yet")
        # else:
        #     outdir = self.outdir

        # parse_rawname

        res = [
            MOKAPOT,
            "--decoy_prefix",
            "rev_",
            "--enzyme",
            self.enzyme,
            # "--missed_cleavages",
            # "2",
            "--dest_dir",
            outdir,
            "--train_fdr",
            self.train_fdr,
            "--test_fdr",
            self.test_fdr,
            "--seed",
            self.seed,
            "--folds",
            self.folds,
            # "--max_iter", 30,
            # "--semi",
            "-v",
            2,
            *[str(pinfile.resolve()) for pinfile in filter(None, pinfiles)],
        ]
        if file_root is not None:
            res.append("--file_root")
            res.append(file_root)

        self._CMD = res
        return self._CMD


class FileMover(Receiver):  # receiver
    NAME = "FileMover Receiver"
    """execute something internally"""

    def run(self, inputfiles=None, outdir=None, CMD=None, **kws) -> Iterable[Path]:
        # move files command

        logger.info(f"starting {self}")
        if outdir is None:
            outdir = Path(".")
            return inputfiles
        logger.info(f"Outdir set to {outdir}")

        newfiles = list()

        for inputfile in inputfiles:
            logger.info(f"{inputfile} --> {newfile}")
            newfile = inputfile.rename(outdir / inputfile.name)
            newfiles.append(newfile)

        return newfiles


# this was fixed previously?
# REGX = "(.*[f|F]?\\d)(?=\\.\\d+\\.\\d+\\.\\d+)"
REGX = "(.*\\d)(?=\\.\\d+\\.\\d+\\.\\d+)"
# this doesn't work if there is a fraction named fA ??


def extract_file_from_scan_header(s: pd.Series):
    return s.str.extract(REGX)


class PrepareForiSPEC(Receiver):  # receiver
    NAME = ""

    # def run(self, *args, e2g_qual, e2g_quant, **kwargs):
    def run(
        self,
        *args,
        containers: List[SampleRunContainer] = None,
        force=False,
        label="none",
        **kwargs,
    ):
        # if "force" in kwargs:
        #     force = kwargs.pop("force")

        if containers is None:
            logger.error(f"no sampleruncontainers passed")
            # this is bad
            return
        containers = [
            container
            for container in containers
            if isinstance(container, SampleRunContainer)
        ]
        # if len(containers) == 0:
        #     logger.error(f"no sampleruncontainers passed")
        #     # this is bad
        #     return

        for container in containers:
            if not isinstance(container, SampleRunContainer):
                continue  # wrong container

            e2g_qual = container.get_file("e2g_QUAL")
            e2g_quant = container.get_file("e2g_QUANT")

            if e2g_qual is None or e2g_quant is None:
                logger.debug(f"missing e2g file for {container}")
                continue

            # e2g_qual = runcontainer.get_file("e2g_QUAL")
            # e2g_quant = runcontainer.get_file("e2g_QUANT")

            # we are still in the for loop iterating over containers
            # TODO be smart, don't just count 9 characters
            # the proper name is rec_run_search_label_e2g.tab
            # we do not have ready access to label - we will just put none (LF) for now
            outf = e2g_qual.parent / Path(f"{e2g_qual.name[:9]}_{label}_e2g.tab")
            if outf.exists() and not force:
                logger.info(f"{outf} already exists")
                continue  # already done
                # return

            df_ql = pd.read_table(e2g_qual)

            df_ql = df_ql[[x for x in df_ql if x != "LabelFLAG"]]
            df_qt = pd.read_table(e2g_quant)
            df = pd.merge(
                df_qt,
                df_ql,
                on=["EXPRecNo", "EXPRunNo", "EXPSearchNo", "GeneID", "SRA"],
            )
            _d = {x: f"e2g_{x}" for x in df.columns}
            _d["LabelFLAG"] = "e2g_EXPLabelFLAG"
            df = df.rename(columns=_d)

            logger.info(f"Writing {outf}")
            df.to_csv(outf, index=False, sep="\t")


class MSPC_Rename(Receiver):  # receiver
    """
    class to clean up file header and values for grouping on 01
    """

    NAME = "01 Cleaner"

    # def run(self, inputfiles=None, outdir=None, CMD=None, **kws) -> Iterable[Path]:
    def run(self, runcontainer=None, **kws):
        logger.info(f"starting {self}")
        if runcontainer is None:
            logger.info(f"{self}: no run container")
            return

        mspcfile = runcontainer.get_file("MSPCRunner")
        if mspcfile is None:
            logger.info(f"Nothing to do. No MSPCRunner files found for {runcontainer}")
            return

        logger.info(f"Starting file cleanup for {mspcfile}")
        df = pd.read_table(mspcfile)
        df.rename(
            columns={
                "scannum": "First Scan",
                "parent_charge": "Charge",
                "mokapot q-value": "q_value",
                "hyperscore": "Hyperscore",
                "deltaRank1Rank2Score": "DeltaScore",
                "Calibrated Observed M/Z": "mzDa",
                "tot_num_ions": "TotalIons",
                "hit_rank": "Rank",
            }
        )
        # fix specid

        regx = "(.*[f|F]?\\d)(?=\\.\\d+\\.\\d+\\.\\d+)"
        df["SpectrumFile"] = extract_file_from_scan_header(df["SpecId"])
        logger.info(f"rewriting {mspcfile}")
        df.to_csv(mspcfile, sep="\t", index=False)

        # if outdir is None:
        #    outdir = Path(".")
        #    return inputfiles
        # logger.info(f"Outdir set to {outdir}")

        #    return inputfiles


class AddPhosBoolean:  # receiver
    """
    Add phospho boolean to a pin file

    """

    NAME = "AddPhosBoolean"

    def run(self, runcontainer=None, **kws):
        logger.info(f"starting {self}")

        file = runcontainer.get_file("pinfile")
        if file is None:
            logger.info(f"Nothing to do. No pinfile found for {runcontainer}")
            return

        from mokapot import read_pin

        if isinstance(file, Path):
            file = str(file)

        df = read_pin(file, to_df=True)

        df["Phos"] = 0
        df.loc[df["Peptide"].str.contains("[79.966", regex=False), "Phos"] = 1
        df = df[[x for x in df if x != "Proteins"] + ["Proteins"]]
        df.to_csv(file, sep="\t", index=False)
        print(f"Added phos bool to {file}")


RECRUN_REGEX = re.compile(r"(\d{5})_(\d+)_(\d+)")


class FileRealtor:  # receiver
    """
    find a new home for files
    """

    NAME = "FileRealtor"

    def __init__(self, searchno=None, **kwargs):
        self.searchno = searchno

    def run(
        self,
        runcontainers: Collection[RunContainer] = tuple(),
        outdir: Path = None,
        searchno: int = 7,
        **kwargs,
    ) -> Dict[Path, List[Path]]:
        """
        :inputfiles: Path objects of files to
        """
        if runcontainers is None:
            raise ValueError("no input")
        results = defaultdict(list)

        # print(inputfiles[[x for x in inputfiles.keys()][0]])
        for (
            run_container
        ) in runcontainers:  # id may be equivalent to stem, but doesn't have to be
            recno, runno, searchno = parse_rawname(run_container.stem)
            if self.searchno is not None:
                if searchno != self.searchno:
                    logger.warning("searchno does not match")
            if recno is None:
                logger.info(f"Could not find recno for {run_container.stem}, skipping")
                continue
            if searchno is None:
                searchno = self.searchno
                logger.info(
                    f"Could not find searchno for {run_container.stem}, setting to {searchno}"
                )
            outname = "_".join(filter(None, (recno, runno, str(searchno))))
            if outdir is None:
                _outdir = run_container.rootdir
            else:
                _outdir = outdir
            if _outdir is None:
                logger.error(
                    f"problem in run container construction for {run_container}"
                )

            new_home = _outdir / outname
            # check if new home already made
            if _outdir.resolve().parts[-1] == new_home.parts[-1]:  #  already created
                new_home = _outdir
            if not new_home.exists():
                logger.info(f"Creating {new_home}")
                new_home.mkdir(exist_ok=False)
            else:
                # logger.info(f"{new_home} already exists.")
                pass

            # logger.info(f"{inputfile} -> {new_home}")

            # move to new home no matter what
            run_container.relocate(new_home)

            results[new_home].append(run_container)
        return runcontainers


class Loop(Command):
    NAME = "loop"

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


from .containers import AbstractContainer


def make_psms_collect_object(container_cls, name=None, path=None):
    # if not isinstance(container_cls, AbstractContainer):
    if not type(container_cls) == type(AbstractContainer):
        raise ValueError(f"wrong type for {container_cls}")

    collect_psms = PythonCommand(
        FileFinder(),
        # file=rawfiles,
        path=path,
        # outdir=outdir,
        container_obj=container_cls,
        # depth=depth,
        name=name,
    )
    return collect_psms


class AddE2GMetadata(Receiver):
    NAME = "AddE2GMetadata"

    def run(
        self,
        sampleruncontainer: SampleRunContainer = None,
        name: str = None,
        outdir: Path = None,
        force: bool = False,
        **kwargs,
    ):
        logger.info(f"starting {self}")

        if sampleruncontainer is None:
            logger.error(f"no sampleruncontainer passed")
            # raise ValueError("no input")

        qualf = sampleruncontainer.get_file("e2g_QUAL")
        quantf = sampleruncontainer.get_file("e2g_QUANT")
        if qualf is None or quantf is None:
            logger.error(f"no e2g files found for {sampleruncontainer}")
            return
        else:
            logger.info(f"{sampleruncontainer}: found {qualf} and {quantf}")

        qualmat = pd.read_table(qualf)
        quantmat = pd.read_table(quantf)
        quantmat = quantmat.loc[~pd.isna(quantmat.GeneID)]

        mat_wide = quantmat.pivot(
            index="GeneID", columns="LabelFLAG", values="iBAQ_dstrAdj"
        )
        new_colnames = list(map(self.fix_colnames, mat_wide.columns))
        mat_wide = mat_wide.set_axis(tuple(new_colnames), axis=1)

        # e2g = ispec.E2G(
        #     *rec_run_search_list, data_dir=sampleruncontainer.rootdir, only_local=True
        # )
        from bcmproteomics_ext import ispec

        rec_run_search = sampleruncontainer.rec_run_search
        exp = ispec.Experiment(
            sampleruncontainer.record_no,
            sampleruncontainer.run_no,
            sampleruncontainer.search_no,
        )

        metadata = exp.metadata
        if metadata.empty:
            logger.error(f"no metadata found for {sampleruncontainer}")
            return

        if metadata is None or metadata.empty:
            logger.error(f"no metadata found for {sampleruncontainer}")
            return
        assert "EXPLabelFLAG" in metadata.columns
        for x in [
            "EXPRecNo",
            "EXPRunNo",
            "EXPSearchNo",
        ]:
            metadata[x] = metadata[x].astype(str)
        metadata.loc[metadata.EXPLabelFLAG == "131", "EXPLabelFLAG"] = "131N"

        metadata = metadata.loc[metadata.EXPLabelFLAG.isin(mat_wide.columns)]
        metadata = metadata.rename(
            columns={
                "EXPRecNo": "recno",
                "EXPRunNo": "runno",
                "EXPSearchNo": "searchno",
                "EXPLabelFLAG": "label",
            }
        )
        # metadata =
        idx_name = metadata.apply(
            lambda x: str.join(
                "_", [x["recno"], x["runno"], x["searchno"], x["label"]]
            ),
            axis=1,
        )
        metadata = metadata.set_index(idx_name)
        metadata["label"] = pd.Categorical(
            metadata.label, ordered=True, categories=mat_wide.columns
        )
        metadata = metadata.sort_values("label")
        mat_wide = mat_wide.set_axis(metadata.index, axis=1)

        # --

        cmapR = importr("cmapR")

        pandas2ri.activate()
        robjects.r.assign("cdesc", metadata.astype(str))
        robjects.r.assign("cid", metadata.index)

        rdesc = qualmat[
            [
                "GeneID",
                "TaxonID",
                "GeneSymbol",
                "Description",
                "PSMs",
                "PeptideCount",
                "GPGroup",
                "PeptidePrint",
                "EXPRecNo",
                "EXPRunNo",
                "EXPSearchNo",
                "LabelFLAG",
            ]
        ]
        # add GeneID, GeneSymbol, and Description from ensembl

        rdesc.loc[pd.isna(rdesc.Description), "Description"] = ""
        if not rdesc.GeneID.is_unique:
            logger.error(f"GeneID is not unique for {sampleruncontainer}")
            return
        rdesc.index = rdesc["GeneID"].astype(str)
        rdesc = rdesc.loc[mat_wide.index]

        robjects.r.assign("rid", rdesc.index)
        robjects.r.assign("rdesc", rdesc)

        robjects.r.assign("edata", mat_wide)
        robjects.r(
            'my_ds <- new("GCT", mat=as.matrix(edata), rid=rid, cid=cid, rdesc=rdesc, cdesc=cdesc)'
        )

        labeltype = metadata["EXPLabelType"][0]
        robjects.r.assign("rootdir", qualf.parent.__str__())
        robjects.r.assign(
            "filename",
            str.join("_", [sampleruncontainer.rec_run_search, labeltype]) + ".gct",
        )
        robjects.r(
            "cmapR::write_gct(my_ds, file.path(rootdir, filename), appenddim=T, precision=8)"
        )

    @staticmethod
    def fix_colnames(s: str):
        """check and fix if we have
        12345_1_1_12x_X and fix to 12345_1_1_12xX
        fix_colnames("12345_1_1_127_N") -> 12345_1_1_127N
        fix_colnames("12345_1_1_126") -> 12345_1_1_126
        """
        res = s.split("_")
        if len(res) == 5:
            res[3] = res[3] + res[4]
            res = res[:4]
        elif len(res) == 3:
            res[1] = res[1] + res[2]
            res = res[:2]
        ret = str.join("_", res)
        ret = ret.lstrip("TMT_")
        return ret


class AddSiteMetadata(Receiver):
    NAME = "AddSiteMetadata"

    @staticmethod
    def fix_colnames(s: str):
        """check and fix if we have
        12345_1_1_12x_X and fix to 12345_1_1_12xX
        fix_colnames("12345_1_1_127_N") -> 12345_1_1_127N
        fix_colnames("12345_1_1_126") -> 12345_1_1_126
        """
        logger.info(f"fixing {s}")
        res = s.split("_")
        if len(res) == 5:
            res[3] = res[3] + res[4]
            res = res[:4]
        elif len(res) == 3:
            res[1] = res[1] + res[2]
            res = res[:3]
        ret = str.join("_", res)
        logger.info(f"returning {ret}")
        return ret

    def run(
        self,
        sampleruncontainer: SampleRunContainer = None,
        name: str = None,
        outdir: Path = None,
        force: bool = False,
        **kwargs,
    ):
        logger.info(f"starting {self}")
        from rpy2.robjects import pandas2ri # ??
        pandas2ri.activate()

        if sampleruncontainer is None:
            logger.error(f"no sampleruncontainer passed")
            # raise ValueError("no input")

        site_table_nr = sampleruncontainer.get_file("site_table_nr")  # Path or None

        if site_table_nr is None:
            logger.error(f"no site table found for {sampleruncontainer}")
            return

        from bcmproteomics_ext import ispec

        recno = sampleruncontainer.record_no
        rec_run_search = sampleruncontainer.rec_run_search
        ##

        rec_run_search_list = rec_run_search.split("_")
        exp = ispec.Experiment(
            sampleruncontainer.record_no,
            sampleruncontainer.run_no,
            sampleruncontainer.search_no,
        )
        # genotype = exp.genotype
        # treatment = exp.treatment
        # description = exp.description

        conf = get_conf()

        # Retrieve the target file path from the config object
        fa_db = None  # we try to load data into fa_db via previously defined parameters
        target_file_name = "GENCODE_hsmm_new"
        target_file_path = conf["refdb"].get(target_file_name, "")
        # Check if the target file is present
        if target_file_path and os.path.exists(target_file_path):
            # Load the file or perform any other operation you need
            # all of this is made up - theoretical but not possible yet
            # chatGPT-4 Aug3
            # will the AI from one generative language model
            # be able to talk to the AI from another generative language model?
            # as I try to think of a good answer, I get new suggestions from GitHub Copilot
            # as it tries to complete my thoughts and make a final judgement.
            # <<< the below is an example >>>
            # so far, it has been able to complete my thoughts and make a final judgement.
            # <<< the above is an example >>>.

            # with open(target_file_path, "r") as file:
            #     content = file.read()
            #     # Add the content or any other processing you need
            #     self.sites[site][target_file_name] = content

            # now for the real code
            _gen = fasta_dict_from_file(target_file_path)  # a generator
            fa_db = pd.DataFrame.from_dict(_gen)

        # load gencode ref db if present

        metadata = exp.metadata
        if metadata is None or metadata.empty:
            logger.error(f"no metadata found for {sampleruncontainer}")
            return
        assert "EXPLabelFLAG" in metadata.columns
        for x in [
            "EXPRecNo",
            "EXPRunNo",
            "EXPSearchNo",
        ]:
            metadata[x] = metadata[x].astype(str)
        metadata.loc[metadata.EXPLabelFLAG == "131", "EXPLabelFLAG"] = "131N"
        # metadata = metadata.loc[:, ~metadata.columns.duplicated("first")]

        # from cmapPy.pandasGEXpress.parse import parse, parse_gct, parse_gctx
        # from cmapPy.pandasGEXpress import write_gct
        # from cmapPy.pandasGEXpress.GCToo import GCToo
        #  ===================================================================

        from rpy2.robjects.packages import importr
        from rpy2 import robjects
        from rpy2.robjects import r
        from rpy2.robjects import pandas2ri

        cmapR = importr("cmapR")


        # gct_obj = parse_gct.parse(site_table_nr.__str__())
        gct_obj = cmapR.parse_gctx(site_table_nr.__str__())

        # data_mat = gct_obj.do_slot("mat") # not using
        # problem - row_metadata_df is not a pandas dataframe
        row_metadata_df_robj = gct_obj.do_slot("rdesc")  # heare
        pandas_df = pd.DataFrame()

        for column in row_metadata_df_robj.colnames:
            print(column)
            numpy_array = row_metadata_df_robj.rx2(str(column))
            pandas_df[str(column)] = numpy_array

        row_metadata_df = pandas_df

        #  ===================================================================
        # row_metadata_df = pandas2ri.rpy2py(row_metadata_df_robj)

        if fa_db is not None:
            fa_db_cols = [
                "ENSP",
                "ENST",
                "ENSG",
                "geneid",
                "taxon",
                "symbol",
                "description",
            ]
            if all([x in fa_db.columns for x in row_metadata_df.columns]):
                pass
            else:
                # merged_df = row_metadata_df.merge(fa_db, left_on="GeneID", right_on="geneid",
                #     how="left")
                # merged_df = row_metadata_df.merge(fa_db, left_on="GeneID", right_on="geneid", how="left", validate="one_to_one")
                fa_db_cols = [x for x in fa_db_cols if x not in row_metadata_df.columns]
                if len(fa_db_cols) > 0:
                    merged_df = row_metadata_df.merge(
                        fa_db[fa_db_cols],
                        left_on="Primary_select",
                        right_on="ENSP",
                        how="left",
                    )
                    # this should be done with a proper join
                    assert merged_df.shape[0] == row_metadata_df.shape[0]
                    merged_df.index = merged_df.id
                    row_metadata_df = merged_df
            # assert

            # if row_metadata_df.Si
            # row_metadata_df.index = row_metadata_df.apply(
            #     lambda x: x["GeneSymbol"] + "_" + x["Primary_select"], axis=1
            # )

        # row_metadata_df = pandas2ri.rpy2py(row_metadata_df) # not using
        col_metadata_df = gct_obj.do_slot("cdesc")
        col_metadata_df = pandas2ri.rpy2py(col_metadata_df)
        # row_ids = pandas2ri.rpy2py(gct_obj.do_slot("rid"))
        # col_ids = pandas2ri.rpy2py(gct_obj.do_slot("cid"))
        row_ids = gct_obj.do_slot("rid")
        col_ids = gct_obj.do_slot("cid")

        new_colnames = map(self.fix_colnames, col_ids)
        new_colnames = list(new_colnames)

        # now add metadata
        # format "rec_run_search_label"
        new_colnames = list(new_colnames)

        assert [len(x) == 4 for x in new_colnames]
        if not all(new_colnames == col_ids):
            col_ids = new_colnames
            logger.info("adjust colnames")
            # gct_obj.col_metadata_df.index = new_colnames
        # if len(col_metadata_df)
        # rec_run_search_label_list = gct_obj.col_metadata_df.index.map(

        rec_run_search_label_list = list(map(lambda x: x.split("_"), col_ids))
        new_col_metadata = pd.DataFrame(
            rec_run_search_label_list,
            columns=["recno", "runno", "searchno", "label"],
            index=col_ids,
        )

        res = pd.merge(
            new_col_metadata,
            metadata,
            left_on=["recno", "runno", "searchno", "label"],
            right_on=[
                "EXPRecNo",
                "EXPRunNo",
                "EXPSearchNo",
                "EXPLabelFLAG",
            ],
        )
        # if "EXPRecNo" in res.columns:
        #    res.drop(columns=["EXPRecNo", "EXPRunNo", "EXPSearchNo"], inplace=True)

        res.index = new_col_metadata.index
        if len(res) != len(metadata):
            logger.error(f"metadata merge failed for {sampleruncontainer}")
            return
        new_col_metadata_df = res
        new_col_metadata_df["id"] = new_col_metadata_df.index

        # new_col_metadata_df.astype(str)

        # robjects.r.assign("new_col_metadata_df", new_col_metadata_df.astype(str))

        import rpy2
        from rpy2 import robjects as ro


        # with rpy2.robjects.conversion.localconverter(ro.default_converter + rpy2.robjects.pandas2ri.converter):
        # bug here not properly deactivated/activated elsewhere or previously
        pandas2ri.deactivate()
        pandas2ri.activate()

        if "Exp_Date" in new_col_metadata_df:
            new_col_metadata_df["Exp_Date"] = new_col_metadata_df["Exp_Date"].astype(
                str
            )
        robjects.r.assign("new_col_metadata_df", new_col_metadata_df)
        robjects.r.assign("gct_obj", gct_obj)
        robjects.r.assign("cids", new_col_metadata_df.index)
        robjects.r.assign("cdesc", new_col_metadata_df.astype(str))
        robjects.r.assign("rdesc", row_metadata_df.astype(str))
        robjects.r.assign("rids", row_ids)
        robjects.r("""m <- gct_obj@mat """)
        # rids_r = pandas2ri.py2rpy(rids)
        robjects.r("""rownames(m) <- rids""")
        robjects.r("""colnames(m) <- cids""")

        # robjects.r.assign("rid", new_row_metadata_df.astype(str))
        robjects.r(
            'my_ds <- new("GCT", mat=m, rid=as.character(rids), cid=cids, rdesc=rdesc, cdesc=cdesc)'
        )
        robjects.r.assign("filename", site_table_nr.__str__())
        robjects.r("cmapR::write_gct(my_ds, filename, appenddim=F, precision=8)")

        # melted_robj = cmapR.melt_gct(gct_obj)
        # this takes a really long time
        # we can just save it in r
        # melted_matrix = pandas2ri.rpy2py(melted_robj)
        robjects.r("m <- as.data.frame(my_ds@mat)")
        robjects.r("rownames(m) <- my_ds@rid")
        robjects.r("m <- tibble::rownames_to_column(m, 'id')")

        _filename = site_table_nr.__str__().strip(".gct") + "_mat.tsv"
        robjects.r.assign("filename", _filename)
        robjects.r("readr::write_tsv(m, filename)")

        # robjects.r("melted_obj_nocdesc <- cmapR::melt_gct(my_ds, keep_cdesc=F)")
        # robjects.r("melted_robject <- cmapR::melt_gct(my_ds)")

        # robjects.r(
        #     """melted_df <- cmapR::melt_gct(my_ds, keep_rdesc=F, keep_cdesc=F, suffixes=c('.site', '.sample'))"""
        # )
        # _filename = site_table_nr.__str__().strip(".gct") + "_melted.tsv"
        # robjects.r.assign("filename", _filename)
        # robjects.r("readr::write_tsv(melted_df, filename)")

        pandas2ri.deactivate()

        # robjects.r(
        #     """melted_df_more <- cmapR::melt_gct(my_ds, keep_rdesc=T, keep_cdesc=F, suffixes=c('.site', '.sample'))"""
        # )
        # _filename = site_table_nr.__str__().strip(".gct") + "_melted_manycolumns.tsv"
        # robjects.r.assign("filename", _filename)
        # robjects.r("readr::write_tsv(melted_df, filename)")

        # _colnames_split = [x.split("_") for x in _colnames]
        # robjects.r.assign("new_col_metadata", new_col_metadata_df)

        # out_obj = GCToo(
        #     data_df=data_df,
        #     row_metadata_df=row_metadata_df,
        #     col_metadata_df=new_col_metadata_df,
        # )
        # write_gct.write(out_obj, out_fname=str(site_table_nr) + "_new.gct")

        # import cmapPy

        # # load nr
        # gct_obj = cmapR.parse_gctx(site_table_nr.__str__())
        # gct_obj.get_attrib("cdesc")

        # pass
        # cmapR.p


# class Builder(ABC):
#     """Builder [...]
#
#     [extended_summary]
#
#     Args:
#         ABC ([type]): [description]
#     """
#
#     ...
#
#     @property
#     @abstractmethod
#     def product(self) -> None:
#         pass
#
#
# class Loop(Builder):
#     def __init__(self, collector) -> None:
#         self._collector = collector  # the command with an `execute()` method
#         super().__init__()
#         self.reset()
#
#     def reset(self) -> None:
#         ...
#
#     def build(self, cls):
#
#         worker = Worker()
#         worker.register("finder", self._collector)
#         result_containers = worker.execute("finder")
#         for ix, result_container in enumerate(result_containers):
#             newcls = cls.new(result_container)
#             worker.register(f"{cls}-{ix}", newcls)
#
#
#         1 + 1
#
#     def execute(self):
#         self.build
#
#         # gpgrouper = gpGrouper(
#         #     cmd_runner,
#         #     #inputfiles=searchruncontainer,
#         #     workers=workers,
#         #     name=f"gpgrouper-{ix}",
#         #     refseq=refseq,
#         #     paramfile=Predefined_gpG.default,
#         #     labeltype=labeltype,
#         #     phospho=phospho,
#         # )
#         # worker.register(f"gpgrouper-{ix}", gpgrouper)
#


class RunContainerBuilder(Receiver):
    """gather runs for all experiments"""

    NAME = "RunContainerBuilder"

    def run(
        self,
        containers: List[RunContainer] = None,
        **kwargs,
    ):
        if containers is None:
            logger.error(f"no runcontainers passed")
            # this is bad
            return


class SampleRunContainerBuilder(Receiver):
    """combine multiple fractions of psm files for a given experiment"""

    NAME = "PSM-Collect"

    def run(
        self,
        containers: List[RunContainer] = None,
        outdir: Path = None,
        **kwargs,
    ) -> str:
        """
        We can use this procedure to create SampleRunContainers
        """

        if containers is None:
            logger.error(f"no runcontainers passed")
            # this is bad
            return
            # raise ValueError("No input")

        # logger.debug(f"{self}")
        filegroups = defaultdict(list)
        # for f in sorted(files):
        for container in containers:
            if not isinstance(container, RunContainer):  # SampleRunContainer skip down
                # Grab from sampleruncontainer?
                record_no = container.record_no
                run_no = container.run_no
                search_no = container.search_no
                continue  # wrong container

            # this is where search could be designated
            recrun = find_rec_run(container.stem)
            # print(recrun)
            if not recrun:
                recrun = container.stem[:10]
                logger.warn(f"Could not get group for {container}, using {recrun}")
                # continue

            if recrun:
                group = f"{recrun[0]}_{recrun[1]}"
                # populate all of our "filegroups"
                # filegroups[group].append(mspcfile)
                filegroups[group].append(container)

        record_no = recrun[0]
        run_no = recrun[1]
        # this is broken
        if isinstance(container, SampleRunContainer):
            search_no = container.search_no
        else:
            search_no = 6

        # =========================== SampleRunContainer ===========================
        sampleruncontainers = list()

        # create run containers

        for group, runcontainers in filegroups.items():
            # recrun = find_rec_run(container.stem)
            # if not recrun:  # ..
            #     continue

            # recrun = {find_rec_run(container.stem) for container in runcontainers}

            # silently drops RunContainers that do not have a pin file
            rootdir = filter(None, {container.rootdir for container in runcontainers})
            rootdir = list(rootdir)
            # assert len(recrun) == 1
            # recrun = list(recrun)[0]
            assert len(rootdir) == 1
            rootdir = list(rootdir)[0]
            rootdir = rootdir

            # record_no = group[0]
            # run_no = group[1]
            # rootdir = rootdir

            samplerun = SampleRunContainer(
                name=group,
                rootdir=rootdir,
                runcontainers=runcontainers,
                record_no=record_no,
                run_no=run_no,
                search_no=search_no,
            )

            sampleruncontainers.append(samplerun)
        return sampleruncontainers
        # print(group)

        # # move this?
        # for f in sorted(files):
        #     print(f)
        # print(len(files))
        # df = pd.concat(pd.read_table(f) for f in files)
        # outname = f"{group}_6_psms_all.txt"
        # df.to_csv(outname, sep="\t", index=False)
        # print(f"Wrote {outname}")

        # for samplerun in sampleruncontainers:
        #     samplerun.check_psms_files()
        #     samplerun.concat(force=force)

        # return sampleruncontainers
