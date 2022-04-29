import os
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path
from time import time

from mspcrunner.containers import RunContainer, SampleRunContainer

from .commands import Command, Receiver
from .predefined_params import PREDEFINED_RMD_TEMPLATES
from .logger import get_logger

logger = get_logger(__name__)

BASEDIR = os.path.split(__file__)[0]

# MSPCRunner/src/ext/annotate_site


class AnnotateSite(Command):

    NAME = "AnnotateSite"
    ANNOTATE_SCRIPT = BASEDIR / Path("ext/annotate_site/annotate_protein.py")

    def __init__(
        self,
        receiver: Receiver,
        workers=1,
        outdir=None,
        name=None,
        inputfiles=None,
        container=None,
        containers=None,
        refseq=None,
        **kwargs,
    ):
        if "outdir" in kwargs:
            outdir = kwargs.pop("outdir")
        if "containers" in kwargs:
            kwargs.pop("containers")

        super().__init__(
            receiver,
            outdir=outdir,
            name=name,
            inputfiles=inputfiles,
            container=container,
            containers=containers,
            **kwargs,
        )

        self.workers = workers
        self.refseq = refseq
        self.containers = containers  # are we using this? No?

    def create(self, sampleruncontainers=None, **kwargs):
        if self.containers is not None and sampleruncontainers is not None:  # add
            sampleruncontainers = sampleruncontainers + list(
                filter(None, self.containers)
            )
        elif sampleruncontainers is None and self.containers:
            sampleruncontainers = self.containers

        if sampleruncontainers is None:
            yield self

        d = self.__dict__.copy()
        for kw in kwargs:
            if kw in d:
                d.update(kw, kwargs[kw])
        d["containers"] = sampleruncontainers
        yield type(self)(**d)

    @property
    def CMD(self):

        if self.containers is None:
            return

        self.containers
        ncores = self.workers

        expids = [x.rec_run_search for x in self.containers]
        # "psm_qual"
        psm_files = filter(None, [x.get_file("psm_QUANT") for x in self.containers])
        # import ipdb

        # ipdb.set_trace()

        cmd = [
            f"python",
            self.ANNOTATE_SCRIPT,
            "--combine",
            "--all-genes",
            "--noplot",
            "--nr",
            "--cores",
            ncores,
            "--fasta",
            self.refseq,
            # "--out",
            # OUTNAME,
        ]
        # for psm_file in psm_files:
        for psm_file in expids:
            cmd.append("--psms")
            cmd.append(psm_file)
        # import ipdb

        # ipdb.set_trace()
        return cmd
