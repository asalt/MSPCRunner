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
    # ANNOTATE_SCRIPT = BASEDIR / Path("ext/annotate_site/annotate_protein.py")
    ANNOTATE_SCRIPT = BASEDIR / Path("sitequant/main.py")

    def __init__(
        self,
        receiver: Receiver,
        workers=1,
        outdir=None,
        name=None,
        sampleruncontainers=None,
        refseq=None,
        force=False,
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
            force=force,
            # container=container,
            #containers=containers,
            **kwargs,
        )

        self.workers = workers
        self.refseq = refseq
        # self.containers = containers  # are we using this? No?
        self.sampleruncontainers = sampleruncontainers
        logger.info(f"AnnotateSite: {self.__dict__}")

    def create(self, sampleruncontainers=None, **kwargs):
        d = self.__dict__.copy()
        d.update(kwargs)
        # we do not care about containers atribute
        d["name"] = d.get("name", "executor")
        d['sampleruncontainers'] = sampleruncontainers
        yield type(self)(**d)


    # def create(self, sampleruncontainers=None, **kwargs):
    #     if self.containers is not None and sampleruncontainers is not None:  # add
    #         sampleruncontainers = sampleruncontainers + list(
    #             filter(None, self.containers)
    #         )
    #     elif sampleruncontainers is None and self.containers:
    #         sampleruncontainers = self.containers

    #     if sampleruncontainers is None:
    #         yield self

    #     d = self.__dict__.copy()
    #     for kw in kwargs:
    #         if kw in d:
    #             d.update(kw, kwargs[kw])
    #     d["containers"] = sampleruncontainers
    #     yield type(self)(**d)

    @property
    def CMD(self):

        if self.sampleruncontainers is None:
            return

        valid_containers = list()
        for sampleruncontainer in self.sampleruncontainers:
            if sampleruncontainer.psms_filePath is None:
                logger.info(f"skipping {sampleruncontainer} because no psms file")
                continue
            if sampleruncontainer.get_file("site_table_nr") is not None and not self.force:
                logger.info(
                    f"Found site_table_nr file for {sampleruncontainer}, not regrouping"
                )
                continue
            valid_containers.append(sampleruncontainer)
        if len(valid_containers) == 0:
            logger.info("No valid containers")
            return
        expids = [x.rec_run_search for x in valid_containers]


        ncores = self.workers

        cmd = [
            f"python",
            self.ANNOTATE_SCRIPT,
            # "--combine",
            # "--all-genes",
            # "--noplot",
            # "--nr",
            "--cores",
            ncores,
            "--fasta",
            self.refseq,
            # "--out",
            # OUTNAME,
        ]
        # for psm_file in psm_files:
        # for psm_file in psm_files:
        for expid in expids:  # load with bcmproteomics
            logger.info(f"Adding {expid} to {self}")
            cmd.append("--psms")
            cmd.append(expid)

        return cmd
