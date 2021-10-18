import ctypes
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from enum import Enum
from glob import glob, iglob
from pathlib import Path
from time import time
from typing import Container, List, Optional, Tuple

import progressbar

progressbar.streams.wrap_stderr()
import click
import typer
from ipdb import set_trace

from mspcrunner.gpGrouper import gpGrouper


from .commands import (
    AddPhosBoolean,
    CMDRunner,
    Command,
    FileFinder,
    FileMover,
    FileRealtor,
    MokaPotConsole,
    MokaPotRunner,
    MSPC_Rename,
    PrepareForiSPEC,
    PythonCommand,
    RawObj,
    Worker,
    get_logger,
    make_psms_collect_object,
)
from .config import config_app
from .containers import AbstractContainer, RunContainer, SampleRunContainer
from .MASIC import MASIC
from .Rmd import Rmd
from .MSFragger import MSFragger
from .inhouse_modisite import AnnotateSite
from .predefined_params import (
    PREDEFINED_QUANT_PARAMS,
    PREDEFINED_REFSEQ_PARAMS,
    PREDEFINED_RMD_TEMPLATES,
    PREDEFINED_SEARCH_PARAMS,
    Predefined_gpG,
    Predefined_Quant,
    Predefined_RefSeq,
    Predefined_Search,
    RMD_OUT_FORMAT,
    RMD_TEMPLATES,
)
from .psm_concat import PSM_Concat, SampleRunContainerBuilder
from .psm_merge import PSM_Merger
from .utils import confirm_param_or_exit, find_rec_run

app = typer.Typer()
run_app = typer.Typer(chain=True)
app.add_typer(run_app, name="run")
app.add_typer(
    config_app,
    name="config",
    help="Help",
    short_help="subcommand for setting up application",
)

from .monitor import monitor

app.add_typer(monitor.app, name="monitor", short_help="watch / receive raw files")

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

        # filecontainers = worker._output.get("experiment_finder", None)

        ## ==========================================================
        # {cmd.NAME} : {cmd._receiver.NAME} with {cmd.CMD}
        logger.info(
            f"""
        {cmd.NAME} : {cmd.CMD}
        """
        )
        res = worker.execute(name)
        ## ==========================================================

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
    if path is None and bool(rawfile) is False:
        path = Path(".")

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

    rawfiles = rawfile  # a list of 0 or more
    worker = Worker()

    # outdir = set_outdir(outdir, path)

    if path is None:
        path = Path(".")
    file_finder = FileFinder()
    collect_experiments = PythonCommand(
        file_finder,
        file=rawfiles,
        path=path,
        container_obj=RunContainer,
        outdir=outdir,
        depth=depth,
        name="experiment_finder",
    )
    worker.register("experiment_finder", collect_experiments)
    worker.set_file(rawfiles)
    worker.set_path(path)

    # perhaps we do not need this here anymore
    inputfiles = worker.execute("experiment_finder")

    if dry:
        cmd_runner = CMDRunner_Tester(production_receiver=CMDRunner)
        file_mover = CMDRunner_Tester(production_receiver=FileMover)
        file_realtor = CMDRunner_Tester(production_receiver=FileRealtor)
    else:
        cmd_runner = CMDRunner()
        file_mover = FileMover()
        file_realtor = FileRealtor()

    #
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
def add_phos_boolean():

    ctx = get_current_context()

    cmd_runner = ctx.obj.get("cmd_runner")
    worker = ctx.obj.get("worker")

    for (ix, run_container) in enumerate(
        worker._output.get("experiment_finder", tuple())
    ):
        name = (f"add-phos-boolean-{ix}",)
        cmd = PythonCommand(
            AddPhosBoolean(),
            runcontainer=run_container,
            inputfiles=worker._output.get("experiment_finder"),
            name=name,
        )
        worker.register(name, cmd)


@run_app.command()
def search(
    preset: Predefined_Search = typer.Option(None, case_sensitive=False),
    paramfile: Optional[Path] = typer.Option(None),
    refseq: Predefined_RefSeq = typer.Option(None),
    local_refseq: Optional[Path] = typer.Option(None),
    calibrate_mass: Optional[int] = typer.Option(default=1, min=0, max=2),
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
    if local_refseq is None:
        refseq = PREDEFINED_REFSEQ_PARAMS.get(refseq, refseq)
    else:
        refseq = local_refseq

    inputfiles = worker._output.get("experiment_finder", tuple())
    # input files not dynamically found, but should be

    msfragger = MSFragger(
        receiver=cmd_runner,
        # inputfiles=worker._output.get("experiment_finder"),
        inputfiles=inputfiles,
        # inputfiles=rawfiles,  # we can set this later
        # paramfile=msfragger_conf.absolute(),
        paramfile=paramfile.resolve(),
        ramalloc=ramalloc,
        refseq=refseq,
        name="msfragger-cmd",
    )
    msfragger.set_param("calibrate_mass", calibrate_mass)
    worker.register("msfragger", msfragger)


@run_app.command()
def make_rmd(
    template: RMD_TEMPLATES = typer.Option("TMTreport", "-t", "--template"),
    output_format: RMD_OUT_FORMAT = typer.Option("html", "-f", "--output-format"),
    report_name: str = typer.Option(
        None,
        "-o",
        "--output-name",
        help="Dasename for output. Default is name of the template file.",
    ),
    # title: str = typer.Option(None),
    # output_format: RMD_OUT_FORMAT = typer.Option(None),
):
    """ """
    # import pkg_resources
    # set_trace()
    # print(f"Here : {pkg_resources.resource_dir}")
    # template_file = confirm_param_or_exit(
    #     template, preset=None, PRESET_DICT=PREDEFINED_RMD_TEMPLATES
    # )

    template_file = Path(template.name)
    report_name = report_name or os.path.splitext(template_file.name)[0]

    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    _collector = make_psms_collect_object(
        container_cls=SampleRunContainer, name="collect-e2gs-for-Rmd", path=worker.path
    )
    worker.register(f"collect-e2gs-for-Rmd", _collector)

    outdir = Path(".").absolute()  # can make dynamic later
    rmd = Rmd(
        receiver=cmd_runner,
        outdir=outdir,
        output_format=output_format,
        report_name=report_name,
        template_file=template_file,
        name="Rmd",
    )
    worker.register(rmd.name, rmd)


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
    # collector_name = "psms_collector_for_percolator"
    # psms_collector = make_psms_collect_object(
    #     container_cls=SampleRunContainer, name=collector_name, path=worker.path
    # )
    # worker.register(collector_name, psms_collector)
    psm_obj = PythonCommand(
        SampleRunContainerBuilder(),
        name=f"SampleRunContainerBuilder",
        force=False,
    )
    worker.register(f"build-sampleruncontainers", psm_obj)

    mokapot = MokaPotConsole(
        cmd_runner,
        outdir=None,  # can add later
        train_fdr=train_fdr,
        test_fdr=test_fdr,
        folds=folds,
        decoy_prefix=decoy_prefix,
        name=f"mokapot-parent"
        # outdir=WORKDIR
    )
    worker.register(f"mokapot-parent", mokapot)
    # this is all we need

    # for (ix, run_container) in enumerate(
    #     worker._output.get("experiment_finder", tuple())
    # ):

    #     # THIS does not work when path is not specified at start
    #     file_maybe_exists = run_container.get_file("mokapot-psms")

    #     # run_container.update_files()
    #     # import ipdb

    #     # ipdb.set_trace()
    #     # if run_container.get_file("pinfile") is None:
    #     #     logger.info(f"{file_maybe_exists} does not have associated pin file")
    #     #     continue

    #     if isinstance(file_maybe_exists, Path) and file_maybe_exists.exists():
    #         logger.info(f"{file_maybe_exists} already present")
    #         continue


@run_app.command()
def concat_psms(
    force: Optional[bool] = typer.Option(False),
):
    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    # 'k
    psms_collector = make_psms_collect_object(
        container_cls=SampleRunContainer,
        name="psms-collector-for-concat",
        path=worker.path,
    )
    worker.register(f"psms-collector-for-concat", psms_collector)

    # worker.register(f"collect-psms", psms_collector)
    psm_obj = PythonCommand(
        SampleRunContainerBuilder(),
        name=f"SampleRunContainerBuilder",
    )
    worker.register(f"build-sampleruncontainers", psm_obj)
    psm_obj = PythonCommand(
        PSM_Concat(),
        # runcontainers=worker._output.get("experiment_finder", tuple()),
        # psm_merger,
        name=f"concat-psms-0",
        force=force,
    )
    worker.register(f"concat_psms", psm_obj)

    # for (ix, runcontainer) in enumerate(
    #     worker._output.get("experiment_finder", tuple())
    # ):

    #     # psm_merger = PSM_Merger()
    #     # if run_container

    #     psm_concat = PythonCommand(
    #         PSM_Concat(),
    #         runcontainer=runcontainer,
    #         # psm_merger,
    #         name=f"psm_concat_{ix}",
    #     )
    #     worker.register(f"psm_concat_{ix}", psm_concat)

    # worker.register(f"PSM-Merger", psm_merger)


@run_app.command()
def mspc_rename():
    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    psms_collector = make_psms_collect_object(
        container_cls=RunContainer, name="experiment_finder", path=worker.path
    )
    worker.register(f"collect-psms-for-rename", psms_collector)

    ix = 0
    file_cleaner = PythonCommand(
        MSPC_Rename(),
        # runcontainer=run_container,
        # psm_merger,
        name=f"renamer_{ix}",
    )
    worker.register(f"renamer_{ix}", file_cleaner)

    # for (ix, run_container) in enumerate(
    #     worker._output.get("experiment_finder", tuple())
    # ):

    #     file_cleaner = PythonCommand(
    #         MSPC_Rename(),
    #         runcontainer=run_container,
    #         # psm_merger,
    #         name=f"renamer_{ix}",
    #     )
    #     worker.register(f"renamer_{ix}", file_cleaner)


@run_app.command()
def prepare_ispec_import(
    label: Optional[str] = typer.Option(
        default="none",
    ),
    force: Optional[bool] = typer.Option(False),
):

    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    find_sample_containers = make_psms_collect_object(
        container_cls=SampleRunContainer, name="experiment_finder", path=worker.path
    )
    worker.register(f"collect-e2gs", find_sample_containers)

    file_cleaner = PythonCommand(
        PrepareForiSPEC(),
        # runcontainer=run_container,
        # psm_merger,
        name=f"ispec-renamer",
        label=label,
        force=force,
    )
    worker.register(f"ispec-renamer", file_cleaner)


@run_app.command()
def merge_psms():
    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    for (ix, runcontainer) in enumerate(
        worker._output.get("experiment_finder", tuple())
    ):

        # psm_merger = PSM_Merger()
        # if run_container

        psm_merger = PythonCommand(
            PSM_Merger(),
            runcontainer=runcontainer,
            # psm_merger,
            name=f"psm_concat_{ix}",
        )
        worker.register(f"psm-merger_{ix}", psm_merger)

    # worker.register(f"PSM-Merger", psm_merger)

    # # for (ix, run_container) in enumerate(
    # #     worker._output.get("experiment_finder", tuple())
    # # ):

    # # psm_merger = PSM_Merger()
    # worker._output.get("experiment_finder", tuple())
    # # create SampleRun objects

    # psm_obj = PythonCommand(
    #     PSM_Merger(),
    #     runcontainers=worker._output.get("experiment_finder", tuple()),
    #     # psm_merger,
    #     name=f"concat_psms_0",
    # )
    # worker.register(f"concat_psms_0", psm_obj)


@run_app.command()
def annotate_sites(
    refseq: Predefined_RefSeq = typer.Option(None),
    local_refseq: Optional[Path] = typer.Option(None),
    workers: int = typer.Option(1, help="number of workers (CPU cores) to deploy"),
):
    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    if local_refseq is None:
        refseq = PREDEFINED_REFSEQ_PARAMS.get(refseq, refseq)
    else:
        refseq = local_refseq

    if refseq is None:
        raise ValueError("No refseq specified")

    psms_collector = make_psms_collect_object(
        container_cls=SampleRunContainer, name="experiment_finder", path=worker.path
    )
    worker.register(f"collect-psms", psms_collector)

    # searchruncontainers = worker.execute("collect-psms")

    # for ix, searchruncontainer in enumerate(searchruncontainers):

    # TODO finish: this is incomplete
    site_annotate_factory = AnnotateSite(
        cmd_runner,
        # inputfiles=searchruncontainer,
        workers=workers,
        name=f"AnnotateSite-inhouse",
        refseq=refseq,
    )

    worker.register(f"site_annotate_factory", site_annotate_factory)


@run_app.command()
def gpgroup(
    local_refseq: Optional[Path] = typer.Option(None),
    # paramfile: Optional[Path] = typer.Option(None),
    # preset: Predefined_Search = typer.Option(None, case_sensitive=False),
    workers: Optional[int] = typer.Option(1),
    labeltype: Optional[str] = typer.Option(None),
    refseq: Predefined_RefSeq = typer.Option(None),
    phospho: Optional[bool] = typer.Option(False),
    no_taxa_redistrib: bool = typer.Option(False, "--no-taxa-redistrib", help=""),
):
    ctx = get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    if local_refseq is None:
        refseq = PREDEFINED_REFSEQ_PARAMS.get(refseq, refseq)
    else:
        refseq = local_refseq

    if refseq is None:
        raise ValueError("No refseq specified")

    # jpsms_collector = make_psms_collect_object(
    # j    container_cls=RunContainer, name="make-runcontainers-for-gpg", path=worker.path
    # j)
    # jworker.register(f"make-runcontainers-for-gpg", psms_collector)

    # ==============================
    # psms_collector = make_psms_collect_object(
    #     container_cls=SampleRunContainer, name="psms-collector-for-concat", path=worker.path
    # )
    # worker.register(f"psms-collector-for-concat", psms_collector)

    # # worker.register(f"collect-psms", psms_collector)
    # psm_obj = PythonCommand(
    #     SampleRunContainerBuilder(),
    #     name=f"SampleRunContainerBuilder",
    # )
    # worker.register(f"build-sampleruncontainers", psm_obj)

    # ====
    psms_collector = make_psms_collect_object(
        container_cls=SampleRunContainer, name="collect-psms-for-gpg", path=worker.path
    )
    worker.register(f"psms-collector-for-gpg", psms_collector)

    psm_obj = PythonCommand(
        SampleRunContainerBuilder(),
        name=f"SampleRunContainerBuilder",
        force=False,
    )
    worker.register(f"build-sampleruncontainers", psm_obj)

    gpgrouper = gpGrouper(
        cmd_runner,
        # inputfiles=searchruncontainer,
        workers=workers,
        name=f"gpgrouper-builder",
        refseq=refseq,
        paramfile=Predefined_gpG.default,
        labeltype=labeltype,
        phospho=phospho,
        no_taxa_redistrib=no_taxa_redistrib,
    )

    worker.register(f"gpgrouper-builder", gpgrouper)


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
