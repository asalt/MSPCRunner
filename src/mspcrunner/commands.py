# commands
from ipdb import set_trace
from abc import abstractclassmethod, abstractmethod
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
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Collection

import ipdb
from .worker import Worker
from .file_finder import FileFinder
from .containers import RunContainer, SampleRunContainer

import click

import pandas as pd

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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            shell=False,
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

        for pipe, msg in merge_pipes(stdout=popen.stdout, stderr=popen.stderr):
            logger.info(msg)

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
        receiver: Receiver,
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
        # self.outdir = outdir or Path(".")
        self.outdir = outdir  # let it stay as none if not given
        for k, v in kwargs.items():
            setattr(self, k, v)

        # def set_files(self, inputfiles: dict):
        logger.info(f"Adding inputfiles on {self}")
        self.inputfiles = inputfiles

    def create(self, containers=None, **kws):
        if containers is None:
            yield self
        else:
            for container in containers:
                d = self.__dict__.copy()
                for kw in kws:
                    if kw in d:
                        d.update(kw, kws[kw])
                yield self(**self.__dict__, inputfiles=container)

    def update_inputfiles(self, *objs):
        self.inputfiles = list()

        for obj in objs:
            if isinstance(obj, (RunContainer, SampleRunContainer)):
                self.inputfiles.append(obj)

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


class FileMover:  # receiver

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


class PrepareForiSPEC:  # receiver

    NAME = ""

    # def run(self, *args, e2g_qual, e2g_quant, **kwargs):
    def run(
        self, *args, containers: List[SampleRunContainer] = None, label="none", **kwargs
    ):

        force = False
        if "force" in kwargs:
            force = kwargs.pop("force")

        if containers is None:
            logger.error(f"no sampleruncontainers passed")
            # this is bad
            return
        # filter for correct container type
        containers = [
            container
            for container in containers
            if isinstance(container, SampleRunContainer)
        ]
        if len(containers) == 0:
            logger.error(f"no sampleruncontainers passed")
            # this is bad
            return

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

    def run(
        self,
        runcontainers: Collection[RunContainer] = tuple(),
        outdir: Path = None,
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
            if recno is None:
                logger.info(f"Could not find recno for {run_container.stem}, skipping")
                continue
            if searchno is None:
                searchno = "6"
            outname = "_".join(filter(None, (recno, runno, searchno)))
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


class PythonCommand(Command):

    NAME = "PythonCommand"

    def __init__(self, *args, **kws):
        super().__init__(
            *args, **kws
        )  # this has to be called before we can access "args"

    def create(self, runcontainers=None, sampleruncontainers=None, **kws):
        # from ipdb import set_trace

        # set_trace()
        # if self._receiver.

        if runcontainers is None and sampleruncontainers is None:
            yield self
        else:
            if runcontainers is None:
                runcontainers = tuple()
            if sampleruncontainers is None:
                sampleruncontainers = tuple()

            containers = list(runcontainers) + list(sampleruncontainers)
            d = self.__dict__.copy()

            # quick fix
            if "receiver" not in d and "_receiver" in d:
                d["receiver"] = d["_receiver"]
            # ================
            for kw in kws:
                if kw in d:
                    d.update(kw, kws[kw])
            if "inputfiles" in d:
                inputfiles = d.pop("inputfiles")
                if inputfiles is not None:
                    pass

            yield PythonCommand(**d, containers=containers)

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
        basename=None,
        **kws,
    ):
        """
        base_name basename of file e.g. 12345_x_x
        """

        super().__init__(*args, **kws)
        self.decoy_prefix = decoy_prefix
        self.train_fdr = train_fdr
        self.test_fdr = test_fdr
        self.seed = seed
        self.folds = folds
        self.outdir = outdir
        self.pinfiles = tuple()
        self.basename = basename  # not using
        self._cmd = None

    @property
    def CMD(self):
        # self.announce()

        # for inputfile in self.get_inputfiles():
        # for inputfile in self.inputfiles:

        if self.inputfiles and isinstance(self.inputfiles, RunContainer):
            pinfiles = [self.inputfiles.get_file("pinfile")]
        elif self.inputfiles and not isinstance(self.inputfiles, RunContainer):
            pinfiles = [
                x.get_file("pinfile") for x in self.inputfiles
            ]  # the values are RawFile instances
            # pinfiles = [x for x in pinfiles if x is not None]
        if pinfiles[0] is None:
            raise ValueError(f"no pinfile found for {self.inputfiles}")

        if self.outdir is None and len(pinfiles) == 1:
            outdir = pinfiles[0].parent
        elif self.outdir is not None and len(pinfiles) > 1:
            raise NotImplementedError("Have not resolved multiple file case yet")
        else:
            outdir = self.outdir

        if self.basename is not None:
            file_root = self.basename
        elif len(pinfiles) == 1:
            file_root = pinfiles[0].stem
        elif len(pinfiles) > 1 and self.basename is None:
            raise NotImplementedError("Have not resolved multiple file case")

        # parse_rawname
        res = [
            MOKAPOT,
            "--decoy_prefix",
            "rev_",
            "--missed_cleavages",
            "2",
            "--dest_dir",
            outdir,
            "--file_root",
            f"{file_root}",
            "--train_fdr",
            self.train_fdr,
            "--test_fdr",
            self.test_fdr,
            "--seed",
            self.seed,
            "--folds",
            self.folds,
            *[str(pinfile.resolve()) for pinfile in pinfiles],
        ]

        self._CMD = res
        return self._CMD

    def execute(self):
        "execute"
        # return self._receiver.action("run", self.CMD)
        # self.find_pinfiles()  # should be created by the time this executs
        return self._receiver.run(CMD=self.CMD)


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
#         import ipdb
#
#         ipdb.set_trace()
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
