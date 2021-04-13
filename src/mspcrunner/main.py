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

app = typer.Typer(chain=True)

# from folderstats import get_folderstats
BASEDIR = Path(os.path.split(__file__)[0])

import logging

from .commands import (
    CMDRunner,
    MokaPotRunner,
    RawObj,
    Command,
    MASIC,
    MSFragger,
    Worker,
    RawObj,
    PythonCommand,
    FileMover,
    FileRealtor,
    MokaPotConsole,
)
from .commands import get_logger
from .commands import CMDRunner_Tester

from .predefined_params import Predefined_Search, Predefined_Quant, PREDEFINED_SEARCH_PARAMS, PREDEFINED_QUANT_PARAMS
from .utils import confirm_param_or_exit

# import db
# from db import get_database_conn

# TODO set this via some config file
#WORKDIR = Path("E:\MSPCRunner\processing\tmp")
#PROCESSING_DIR = Path("E:\MSPCRunner\processing")
WORKDIR = Path('.')
PROCESSING_DIR = Path('.')
MASIC_DEFAULT_CONF = Path("masic_default.xml")  # need to have LF, TMT10, 11, 16..
# will change later
MSFRAGGER_DEFAULT_CONF = Path(
    "../params/MSfragger_OTIT_hs.conf"
)  # need to have LF, TMT10, 11, 16..

MASIC_DEFAULT_CONF = Path("../")  # need to have LF, TMT10, 11, 16..

logger = get_logger(__name__)


# move all the database checking logic to central server
def _survey(workdir=WORKDIR):
    if isinstance(workdir, str):
        workdir = Path(workdir)

    for rawfile in workdir.glob("*raw"):

        rawobj = RawObj(rawfile)
        identifiers = rawobj.parse_rawname()
        if len(identifiers) == 2:  # rec and run
            recno, runno = (int(x) for x in identifiers)

        exp_selector = db.Experiment.select().where(recno == recno)

        label, _ = db.Label.get_or_create(name="none")
        if not exp_selector.exists():  # new exp
            experiment = db.Experiment.create(recno=recno, label=label)
            logger.info("Created {experiment}")
        else:
            experiment = exp_selector.first()

        # rec_run = db.Experiment.join(ExpRun)
        if len(experiment.runs) == 0 or not [x.runno == runno for x in experiment.runs]:
            exprun = db.ExpRun.create(runno=runno, recno=experiment)
            logger.info(f"Created {exprun}")
        # TODO handle search
        # if len(experiment.runs) == 0 or not [x.recno=runno for x in experiment.searches]:
        if len(experiment.searches) == 0:  # no run
            searchno = 1
        else:
            searchno = max(x.searchno for x in experiment.searches)
        expsearch = db.ExpSearch.create(searchno=searchno, recno=experiment)
        logger.info(f"Created {expsearch}")

        rawfile_record = db.RawFile.create(
            filename=rawfile.name,
            exprec=experiment,
            exprun=exprun,
            birth=rawfile.stat().st_size,
            size=rawfile.stat().st_ctime,
        )
        logger.info(f"Created {rawfile_record}")

        # rec_run_sel = (Experiment.select().join(ExpRun)
        #            .where((Experiment.recno==recno) & (ExpRun.runno==runno))
        # )
        # rec_run = rec_run_sel.first()

        # do_search(rawfile_record)
        do_search(rawfile.name, CMDRunner=CMDRunner_Tester)


def do_search(*rawfiles, paramf=MASIC_DEFAULT_CONF, CMDRunner=CMDRunner):

    if isinstance(paramf, str):
        paramf = Path(paramf)

    cmd_runner = CMDRunner()
    masic = MASIC(cmd_runner, paramf, *rawfiles, name="alex-test")
    # TODO make msfragger.conf
    msfragger = MSFragger(
        cmd_runner, "msfragger.conf", *rawfiles, ramalloc="20G", name="alex-test"
    )

    worker = Worker()
    worker.register("masic", masic)
    worker.register("msfragger", msfragger)
    worker.execute("masic")
    worker.execute("msfragger")


def worker_run(*args, **kwargs):
    """
    run context.obj['worker'] jobs in order of registration
    """
    logging.info('**worker_run**')
    logger.info("ready to execute")
    ctx = click.get_current_context()
    worker = ctx.obj["worker"]

    for name, cmd in worker._commands.items():
        print(name)
        logger.info(f"""
        {cmd.NAME} : {cmd._receiver.NAME} with {cmd.CMD}
        """)
        res = worker.execute(name)
    #import ipdb; ipdb.set_trace()
    
    #msfragger= worker._commands['msfragger']
    1+1
    return

@app.callback(
    invoke_without_command=True, no_args_is_help=True, result_callback=worker_run
)
def main(
    ctx: typer.Context,
    dry: bool = typer.Option(
        False, "--dry", help="Dry run, do not actually execute commands"
    ),
    path: Optional[Path] = typer.Option(
        default=None, help="Path with raw files to process. Will process all raw files in path."
    ),
    rawfile: Optional[List[Path]] = typer.Option(
        default=None, exists=True, help="raw file to process"
    ),
):
    """
    run MSPC pipeline: raw -> MASIC -> MSFragger -> Percolator
    """
    #ctx = click.get_current_context()

    if ctx.invoked_subcommand is None:
        logger.info("starting MSPCRunner")

    if path:
        rawfiles_in_path = path.glob("*raw")
        rawfile += list(rawfiles_in_path)
    rawfiles = rawfile  # just for semantics

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
        inputfiles=rawfiles,
        outdir=PROCESSING_DIR,
        name="file-folder-mapping",
    )

    worker = Worker()
    worker.register('output_finder', calc_outdirs)

    #stage_files = PythonCommand(
        #file_mover, inputfiles=rawfiles, outdir=PROCESSING_DIR, name="filestager"
    #)
    #worker.register("stage_files", stage_files)

    ctx.obj = dict(
        rawfiles=rawfiles,
        cmd_runner=cmd_runner,
        file_mover=file_mover,
        file_realtor=file_realtor,
        worker=worker,
    )
    
    # worker = Worker()





@app.command()
def search(
    preset: Predefined_Search = typer.Option(None, case_sensitive=False),
    paramfile: Optional[Path] = typer.Option(None),
    ramalloc: Optional[int] = typer.Option(
        default=10, help="Amount of memory (in GB) for msfragger"
    ),
    msfragger_conf: Optional[Path] = typer.Option(MSFRAGGER_DEFAULT_CONF),
):
    logger.info("welcome to search")

    ctx = click.get_current_context()
    rawfiles = ctx.obj['rawfiles']
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    paramfile = confirm_param_or_exit(paramfile, preset, PREDEFINED_SEARCH_PARAMS)

    msfragger = MSFragger(
        cmd_runner,
        inputfiles=rawfiles,  # we can set this later
        #paramfile=msfragger_conf.absolute(),
        paramfile=paramfile.absolute(),
        ramalloc=ramalloc,
        name="msfragger-cmd",
    )
    worker.register("msfragger", msfragger)

@app.command()
def quant(
    preset: Predefined_Quant = typer.Option(None, case_sensitive=False),
    paramfile: Optional[Path] = typer.Option(None),
    #masic_conf: Optional[Path] = typer.Option(MASIC_DEFAULT_CONF),

    ):
    logger.info("welcome to quant")

    ctx = click.get_current_context()
    rawfiles = ctx.obj['rawfiles']
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    paramfile = confirm_param_or_exit(paramfile, preset, PREDEFINED_QUANT_PARAMS)


    masic = MASIC(
        cmd_runner,
        inputfiles=rawfiles,  # we can set this later
        #paramfile=msfragger_conf.absolute(),
        paramfile=paramfile.absolute(),
        #ramalloc=ramalloc,
        name="masic-cmd",
    )
    worker.register("masic", masic)



@app.command()
def percolate(
    train_fdr: Optional[float] = typer.Option(default=0.05, help="train fdr"),
    test_fdr: Optional[float] = typer.Option(default=0.05, help="test fdr"),
    folds: Optional[int] = typer.Option(
        default=3, help="number of cross-validation folds to use"
    ),
    decoy_prefix: Optional[str] = typer.Option(
        default="rev_", help="decoy prefix in database"
    ),
):
    ctx = click.get_current_context()
    cmd_runner = ctx.obj["cmd_runner"]
    worker = ctx.obj["worker"]

    mokapot = MokaPotConsole(
        cmd_runner,
        train_fdr=train_fdr,
        test_fdr=test_fdr,
        folds=folds,
        decoy_prefix=decoy_prefix,
    )
    worker.register("mokapot", mokapot)



@app.command()
def something_completely_different():
    return


@app.command()
def run(
    dry: bool = typer.Option(
        False, "--dry", help="Dry run, do not actually execute commands"
    ),
    fasta_db: Optional[Path] = typer.Option(default=Path("test.fa")),
    masic_conf: Optional[Path] = typer.Option(MASIC_DEFAULT_CONF),
    msfragger_conf: Optional[Path] = typer.Option(MSFRAGGER_DEFAULT_CONF),
    rawfiles: List[Path] = typer.Argument(..., exists=True),
):
    """
    run MSPC pipeline: raw -> MASIC -> MSFragger -> Percolator

    """

    logger.info("Starting search")

    # receivers
    if dry:
        cmd_runner = CMDRunner_Tester(production_receiver=CMDRunner)
        file_mover = CMDRunner_Tester(production_receiver=FileMover)
        file_realtor = CMDRunner_Tester(production_receiver=FileRealtor)
    else:
        cmd_runner = CMDRunner()
        file_mover = FileMover()
        file_realtor = FileRealtor()

    worker = Worker()

    calc_outdirs = PythonCommand(
        file_realtor,
        inputfiles=rawfiles,
        outdir=PROCESSING_DIR,
        name="file-folder-mapping",
    )

    # stage_files = PythonCommand(
    #     file_mover, inputfiles=rawfiles, outdir=PROCESSING_DIR, name="filestager"
    # )
    # worker.register("stage_files", stage_files)
    # staged_rawfiles = worker.execute("stage_files")
    # import ipdb; ipdb.set_trace()

    # masic = MASIC(cmd_runner, inputfiles=staged_rawfiles, paramfile=masic_conf, outdir=PROCESSING_DIR)
    # worker.register("masic", masic)
    msfragger = MSFragger(
        cmd_runner,
        inputfiles=rawfiles,
        paramfile=msfragger_conf.absolute(),
        ramalloc=ramalloc,
        name="msfragger-cmd",
    )
    worker.register("msfragger", msfragger)

    # mokapot_cmd = MokaPotRunner(cmd_runner, input_files'*pin files calculate')

    # worker.execute("masic")
    worker.execute("msfragger")

    results = PROCESSING_DIR.glob
    route_files = PythonCommand(
        file_mover,
    )

    logger.info("End of search routine\n")


@app.command()
def test():

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

    ## Receiver
    # cmd_runner = CMDRunner()

    # masic = MASIC(cmd_runner, paramf, *inputfiles)

    ## cmd
    # masic = MASIC(cmd_runner, paramf, *inputfiles)

    # worker = Worker()
    # worker.register("masic", masic)
    # worker.execute("masic")
    ##  masic = MASIC(cmd_runner)
    ##  worker.register('masic', masic)

    ##FullPipelineWorker

    ##mascic_runner = MASIC(CMDRunner, paramf, *inputfiles)


if __name__ == "__main__":
    app()
    # test()
    # survey()

    # inputfiles = ["a.raw", "b.raw"]
    # paramf = "paramf.xml"

    ## Receiver
    # cmd_runner = CMDRunner()

    # masic = MASIC(cmd_runner, paramf, *inputfiles)

    ## cmd
    # masic = MASIC(cmd_runner, paramf, *inputfiles)

    # worker = Worker()
    # worker.register("masic", masic)
    # worker.execute("masic")
    ##  masic = MASIC(cmd_runner)
    ##  worker.register('masic', masic)

    ##FullPipelineWorker

    ##mascic_runner = MASIC(CMDRunner, paramf, *inputfiles)
