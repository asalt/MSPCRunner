from .containers import RunContainer
from .utils import confirm_param_or_exit
from .predefined_params import (
    Predefined_Search,
    Predefined_Quant,
    Predefined_RefSeq,
    PREDEFINED_SEARCH_PARAMS,
    PREDEFINED_REFSEQ_PARAMS,
    PREDEFINED_QUANT_PARAMS,
)
from .commands import CMDRunner_Tester
from .commands import get_logger
from .MASIC import MASIC
from .MSFragger import MSFragger
from .commands import (
    CMDRunner,
    FileFinder,
    MokaPotRunner,
    RawObj,
    Command,
    Worker,
    RawObj,
    PythonCommand,
    FileMover,
    FileRealtor,
    MokaPotConsole,
)
from .psm_merge import PSM_Merger
from .psm_concat import PSM_Concat
import logging
import sys
import os
import re
from enum import Enum
from pathlib import Path
from glob import iglob, glob
from time import time
import subprocess
import shutil
import ctypes
from typing import Optional, List, Tuple

import click
import typer

from .config import config_app

app = typer.Typer()
run_app = typer.Typer(chain=True)
app.add_typer(run_app, name="run")
app.add_typer(
    config_app,
    name="config",
    help="Help",
    short_help="subcommand for setting up application",
)


# from folderstats import get_folderstats
BASEDIR = Path(os.path.split(__file__)[0])
# import db
# from db import get_database_conn

# TODO set this via some config file
# WORKDIR = Path("E:\MSPCRunner\processing\tmp")
# PROCESSING_DIR = Path("E:\MSPCRunner\processing")
# WORKDIR = Path('.')
PROCESSING_DIR = Path(".")
# need to have LF, TMT10, 11, 16..
MASIC_DEFAULT_CONF = Path("masic_default.xml")
# will change later
MSFRAGGER_DEFAULT_CONF = Path(
    "../params/MSFragger_OTIT_hs.conf"
)  # need to have LF, TMT10, 11, 16..

MASIC_DEFAULT_CONF = Path("../")  # need to have LF, TMT10, 11, 16..

logger = get_logger(__name__)


class Context:

    _worker = None
    _experiments = None

    def __init__(
        self,
    ):
        self.ctx = dict()


def make_new_context():
    return Context()


def get_current_context():
    """
    can make our own context
    """

    ctx = click.get_current_context(silent=True)
    if ctx is None:
        ctx = make_new_context()

    return ctx


# move all the database checking logic to central server


def set_outdir(outdir, path):
    # TODO add test
    if outdir is None and path is None:
        # outdir = Path(os.path.abspath(os.path.dirname(sys.argv[0])))
        outdir = Path(os.getcwd())
    elif outdir is None and path is not None:
        outdir = path
    return outdir


# def _survey(workdir=WORKDIR):
#     if isinstance(workdir, str):
#         workdir = Path(workdir)
#
#     for rawfile in workdir.glob("*raw"):
#
#         rawobj = RawObj(rawfile)
#         identifiers = rawobj.parse_rawname()
#         if len(identifiers) == 2:  # rec and run
#             recno, runno = (int(x) for x in identifiers)
#
#         exp_selector = db.Experiment.select().where(recno == recno)
#
#         label, _ = db.Label.get_or_create(name="none")
#         if not exp_selector.exists():  # new exp
#             experiment = db.Experiment.create(recno=recno, label=label)
#             logger.info("Created {experiment}")
#         else:
#             experiment = exp_selector.first()
#
#         # rec_run = db.Experiment.join(ExpRun)
#         if len(experiment.runs) == 0 or not [x.runno == runno for x in experiment.runs]:
#             exprun = db.ExpRun.create(runno=runno, recno=experiment)
#             logger.info(f"Created {exprun}")
#         # TODO handle search
#         # if len(experiment.runs) == 0 or not [x.recno=runno for x in experiment.searches]:
#         if len(experiment.searches) == 0:  # no run
#             searchno = 1
#         else:
#             searchno = max(x.searchno for x in experiment.searches)
#         expsearch = db.ExpSearch.create(searchno=searchno, recno=experiment)
#         logger.info(f"Created {expsearch}")
#
#         rawfile_record = db.RawFile.create(
#             filename=rawfile.name,
#             exprec=experiment,
#             exprun=exprun,
#             birth=rawfile.stat().st_size,
#             size=rawfile.stat().st_ctime,
#         )
#         logger.info(f"Created {rawfile_record}")
#
#         # rec_run_sel = (Experiment.select().join(ExpRun)
#         #            .where((Experiment.recno==recno) & (ExpRun.runno==runno))
#         # )
#         # rec_run = rec_run_sel.first()
#
#         # do_search(rawfile_record)
#         do_search(rawfile.name, CMDRunner=CMDRunner_Tester)
#
#
# def do_search(*rawfiles, paramf=MASIC_DEFAULT_CONF, CMDRunner=CMDRunner):
#
#     if isinstance(paramf, str):
#         paramf = Path(paramf)
#
#     cmd_runner = CMDRunner()
#
#     masic = MASIC(cmd_runner, paramf, *rawfiles, name="alex-test")
#     # TODO make msfragger.conf
#     msfragger = MSFragger(
#         cmd_runner, "msfragger.conf", *rawfiles, ramalloc="20G", name="alex-test"
#     )
#
#     worker = Worker()
#     worker.register("masic", masic)
#     worker.register("msfragger", msfragger)
#     worker.execute("masic")
#     worker.execute("msfragger")
#


def worker_run(*args, **kwargs):
    """
    run context.obj['worker'] jobs in order of registration
    """
    logger.info("**worker_run**")
    logger.info("ready to execute")
    ctx = get_current_context()
    worker = ctx.obj["worker"]

    for name, cmd in worker._commands.items():
        print(name)
        # filecontainers = worker._output.get("experiment_finder", None)

        logger.info(
            f"""
        {cmd.NAME} : {cmd._receiver.NAME} with {cmd.CMD}
        """
        )
        res = worker.execute(name)
        # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()

    # msfragger= worker._commands['msfragger']
    1 + 1
    return


@run_app.callback(
    invoke_without_command=True, no_args_is_help=True, result_callback=worker_run
)
def main(
    ctx: typer.Context,
    dry: bool = typer.Option(
        False, "--dry", help="Dry run, do not actually execute commands"
    ),
    path: Optional[Path] = typer.Option(
        default=None,
        help="Path with raw files to process. Will process all raw files in path.",
    ),
    outdir: Optional[Path] = typer.Option(
        default=None,
        help="Root directory to store results. Defaults to current directory.",
    ),
    depth: Optional[int] = typer.Option(
        default=2,
        help="recursion depth for rawfile search. to be used with `path` argument",
    ),
    rawfile: Optional[List[Path]] = typer.Option(
        default=None, exists=True, help="raw file to process"
    ),
):
    """
    run MSPC pipeline: raw -> MASIC -> MSFragger -> Percolator
    """
    # ctx = get_current_context()

    if ctx.invoked_subcommand is None:
        logger.info("starting MSPCRunner")

    # TODO look for raw and mzML
    # TODO replace with PythonCommand with FileFinder receiver
    # if path:
    #     for i in range(depth):
    #         #_glob = '*/'*i + '*mzML'
    #         _glob = '*/'*i + '*raw'
    #         rawfiles_in_path = path.glob(_glob)
    #         for f in rawfiles_in_path:
    #             if f.is_file():
    #                 rawfile.append(f)
    # rawfiles = rawfile  # just for semantics
    # print(rawfiles)
    # import ipdb; ipdb.set_trace()

    worker = Worker()

    outdir = set_outdir(outdir, path)

    file_finder = FileFinder()
    collect_experiments = PythonCommand(
        file_finder,
        file=rawfile,
        path=path,
        depth=depth,
        name="experiment_finder",
    )
    worker.register("experiment_finder", collect_experiments)
    inputfiles = worker.execute("experiment_finder")

    if dry:
        cmd_runner = CMDRunner_Tester(production_receiver=CMDRunner)
        file_mover = CMDRunner_Tester(production_receiver=FileMover)
        file_realtor = CMDRunner_Tester(production_receiver=FileRealtor)
    else:
        cmd_runner = CMDRunner()
        file_mover = FileMover()
        file_realtor = FileRealtor()

    calc_outdirs = PythonCommand(
        file_realtor,
        inputfiles=inputfiles,
        # inputfiles=rawfiles,
        # outdir=PROCESSING_DIR,
        outdir=outdir,
        name="file-folder-mapping",
    )

    worker.register("output_finder", calc_outdirs)

    # stage_files = PythonCommand(
    # file_mover, inputfiles=rawfiles, outdir=PROCESSING_DIR, name="filestager"
    # )
    # worker.register("stage_files", stage_files)

    ctx.obj = dict(
        # rawfiles=rawfiles,
        cmd_runner=cmd_runner,
        file_mover=file_mover,
        file_realtor=file_realtor,
        worker=worker,
    )

    # worker = Worker()


@run_app.command()
def search(
    preset: Predefined_Search = typer.Option(None, case_sensitive=False),
    paramfile: Optional[Path] = typer.Option(None),
    refseq: Predefined_RefSeq = typer.Option(None),
    ramalloc: Optional[int] = typer.Option(
        default=10, help="Amount of memory (in GB) for msfragger"
    ),
    # msfragger_conf: Optional[Path] = typer.Option(MSFRAGGER_DEFAULT_CONF),
):
    logger.info("welcome to search")

    ctx = get_current_context()
    # rawfiles = ctx.obj.get("rawfiles")
    cmd_runner = ctx.obj.get("cmd_runner")
    worker = ctx.obj.get("worker")

    paramfile = confirm_param_or_exit(paramfile, preset, PREDEFINED_SEARCH_PARAMS)
    refseq = PREDEFINED_REFSEQ_PARAMS.get(refseq, refseq)

    msfragger = MSFragger(
        cmd_runner,
        inputfiles=worker._output.get("experiment_finder"),
        # inputfiles=rawfiles,  # we can set this later
        # paramfile=msfragger_conf.absolute(),
        paramfile=paramfile.resolve(),
        ramalloc=ramalloc,
        refseq=refseq,
        name="msfragger-cmd",
    )
    worker.register("msfragger", msfragger)


@run_app.command()
def quant(
    preset: Predefined_Quant = typer.Option(None, case_sensitive=False),
    paramfile: Optional[Path] = typer.Option(None),
    # masic_conf: Optional[Path] = typer.Option(MASIC_DEFAULT_CONF),
):
    logger.info("welcome to quant")

    ctx = get_current_context()
    # rawfiles = ctx.obj["rawfiles"]

    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    paramfile = confirm_param_or_exit(paramfile, preset, PREDEFINED_QUANT_PARAMS)

    for (ix, run_container) in enumerate(
        worker._output.get("experiment_finder", tuple())
    ):

        masic = MASIC(
            cmd_runner,
            inputfile=run_container,
            # inputfiles=worker._output.get("experiment_finder"),
            # inputfile=run_container,
            # paramfile=msfragger_conf.absolute(),
            paramfile=paramfile.absolute(),
            # ramalloc=ramalloc,
            name=f"masic-cmd-{ix}",
        )
        worker.register(f"masic-cmd-{ix}", masic)


@run_app.command()
def percolate(
    train_fdr: Optional[float] = typer.Option(default=0.01, help="train fdr"),
    test_fdr: Optional[float] = typer.Option(default=0.01, help="test fdr"),
    folds: Optional[int] = typer.Option(
        default=3, help="number of cross-validation folds to use"
    ),
    decoy_prefix: Optional[str] = typer.Option(
        default="rev_", help="decoy prefix in database"
    ),
):

    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    # rawfiles = ctx.obj["rawfiles"]

    # for ix, rawfile in enumerate(rawfiles):

    for (ix, run_container) in enumerate(
        worker._output.get("experiment_finder", tuple())
    ):

        file_maybe_exists = run_container.get_file("mokapot-psms")

        if run_container.get_file("pinfile") is None:
            logger.info(f"{file_maybe_exists} does not have associated pin file")
            continue

        if isinstance(file_maybe_exists, Path) and file_maybe_exists.exists():
            logger.info(f"{file_maybe_exists} already present")
            continue

        mokapot = MokaPotConsole(
            cmd_runner,
            # inputfiles=(rawfile,),
            # outdir=rawfile.parent.resolve(),
            # inputfiles=worker._output.get("experiment_finder"),
            inputfiles=run_container,
            train_fdr=train_fdr,
            test_fdr=test_fdr,
            folds=folds,
            decoy_prefix=decoy_prefix,
            name=f"mokapot-{ix}"
            # outdir=WORKDIR
        )
        worker.register(f"mokapot-{ix}", mokapot)


@run_app.command()
def merge_psms():
    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    for (ix, run_container) in enumerate(
        worker._output.get("experiment_finder", tuple())
    ):

        # psm_merger = PSM_Merger()
        # if run_container

        merge_psms = PythonCommand(
            PSM_Merger(),
            runcontainer=run_container,
            # psm_merger,
            name=f"merge_psms_{ix}",
        )
        worker.register(f"merge_psms_{ix}", merge_psms)

        # worker.register(f"PSM-Merger", psm_merger)


@run_app.command()
def concat_psms():
    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    # for (ix, run_container) in enumerate(
    #     worker._output.get("experiment_finder", tuple())
    # ):

    # psm_merger = PSM_Merger()

    concat_psms = PythonCommand(
        PSM_Concat(),
        runcontainers=worker._output.get("experiment_finder", tuple()),
        # psm_merger,
        name=f"concat_psms_0",
    )
    worker.register(f"concat_psms_0", concat_psms)


# @app.command()
# def run(
#     dry: bool = typer.Option(
#         False, "--dry", help="Dry run, do not actually execute commands"
#     ),
#     fasta_db: Optional[Path] = typer.Option(default=Path("test.fa")),
#     masic_conf: Optional[Path] = typer.Option(MASIC_DEFAULT_CONF),
#     msfragger_conf: Optional[Path] = typer.Option(MSFRAGGER_DEFAULT_CONF),
#     rawfiles: List[Path] = typer.Argument(..., exists=True),
# ):
#     """
#     run MSPC pipeline: raw -> MASIC -> MSFragger -> Percolator
#
#     """
#
#     logger.info("Starting search")
#
#     # receivers
#     if dry:
#         cmd_runner = CMDRunner_Tester(production_receiver=CMDRunner)
#         file_mover = CMDRunner_Tester(production_receiver=FileMover)
#         file_realtor = CMDRunner_Tester(production_receiver=FileRealtor)
#     else:
#         cmd_runner = CMDRunner()
#         file_mover = FileMover()
#         file_realtor = FileRealtor()
#
#     worker = Worker()
#
#     calc_outdirs = PythonCommand(
#         file_realtor,
#         inputfiles=rawfiles,
#         outdir=PROCESSING_DIR,
#         name="file-folder-mapping",
#     )
#
#     # stage_files = PythonCommand(
#     #     file_mover, inputfiles=rawfiles, outdir=PROCESSING_DIR, name="filestager"
#     # )
#     # worker.register("stage_files", stage_files)
#     # staged_rawfiles = worker.execute("stage_files")
#     # import ipdb; ipdb.set_trace()
#
#     # masic = MASIC(cmd_runner, inputfiles=staged_rawfiles, paramfile=masic_conf, outdir=PROCESSING_DIR)
#     # worker.register("masic", masic)
#     msfragger = MSFragger(
#         cmd_runner,
#         inputfiles=rawfiles,
#         paramfile=msfragger_conf.absolute(),
#         ramalloc=ramalloc,
#         name="msfragger-cmd",
#     )
#     worker.register("msfragger", msfragger)
#
#     # mokapot_cmd = MokaPotRunner(cmd_runner, input_files'*pin files calculate')
#
#     # worker.execute("masic")
#     worker.execute("msfragger")
#
#     results = PROCESSING_DIR.glob
#     route_files = PythonCommand(
#         file_mover,
#     )
#
#     logger.info("End of search routine\n")
#


@run_app.command()
def test():
    """do not use"""

    cmd_runner = CMDRunner_Tester()

    # masic = MASIC_Tester(cmd_runner, "paramf.xml", "a.raw", "b.raw")
    # masic = MASIC_Tester(cmd_runner, "paramf.xml", "a.raw", "b.raw")
    # cmd_runner.register("masic", masic)

    masic = MASIC(cmd_runner, "paramf.xml", "a.raw", "b.raw")
    msfragger = MSFragger(
        cmd_runner,
        paramfile="msfragger.conf",
        inputfiles=["a.raw", "b.raw"],
        ramalloc="20G",
    )

    worker = Worker()
    worker.register("masic", masic)
    worker.register("msfragger", msfragger)
    worker.execute("masic")
    worker.execute("msfragger")


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        logger.error(e.args)
    logger.info("Complete\n")
    # test()
    # survey()

    # inputfiles = ["a.raw", "b.raw"]
    # paramf = "paramf.xml"

    # Receiver
    # cmd_runner = CMDRunner()

    # masic = MASIC(cmd_runner, paramf, *inputfiles)

    # cmd
    # masic = MASIC(cmd_runner, paramf, *inputfiles)

    # worker = Worker()
    # worker.register("masic", masic)
    # worker.execute("masic")
    ##  masic = MASIC(cmd_runner)
    ##  worker.register('masic', masic)

    # FullPipelineWorker

    ##mascic_runner = MASIC(CMDRunner, paramf, *inputfiles)


if __name__ == "__main__":
    app()
    # test()
    # survey()

    # inputfiles = ["a.raw", "b.raw"]
    # paramf = "paramf.xml"

    # Receiver
    # cmd_runner = CMDRunner()

    # masic = MASIC(cmd_runner, paramf, *inputfiles)

    # cmd
    # masic = MASIC(cmd_runner, paramf, *inputfiles)

    # worker = Worker()
    # worker.register("masic", masic)
    # worker.execute("masic")
    ##  masic = MASIC(cmd_runner)
    ##  worker.register('masic', masic)

    # FullPipelineWorker

    ##mascic_runner = MASIC(CMDRunner, paramf, *inputfiles)
