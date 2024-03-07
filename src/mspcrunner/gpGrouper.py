# gpGrouper.py


import logging
import os
import ipdb

from ipdb import set_trace

from .commands import Command, Receiver
from .utils import find_rec_run
from .predefined_params import Predefined_gpG

from .logger import get_logger

logger = get_logger(__name__)

##    "15 17 19",
# _ion_score_bins = [16, 18, 20]
# _ion_score_bins = [15, 17, 19]
#
# _ion_score_bins = "15 17 19"

class gpGrouper(Command):
    NAME = "gpGrouper"

    def __init__(
        self,
        receiver,
        paramfile=None,
        outdir=None,
        name=None,
        refseq=None,
        workers=1,
        force=False,  # if true will run even if output already exists
        runcontainer=None,
        sampleruncontainer=None,
        runcontainers=None,  # either pass 1 run
        sampleruncontainers=None,  # or 1 sample (collection of runs)
        **kwargs,
    ):
        super().__init__(
            receiver,
            paramfile=paramfile,
            name=name,
            refseq=refseq,
            force=force,
            **kwargs,
        )
        # super().__init__(
        #     receiver,
        #     # inputfiles=inputfiles,
        #     paramfile=paramfile,
        #     outdir=outdir,
        #     name=name,
        #     **kwargs,
        # )
        logger.info(f"gpGrouper {self} __init__")
        self.workers = workers
        self.phospho = kwargs.get("phospho", False)
        self.no_taxa_redistrib = kwargs.get("no_taxa_redistrib", False)
        # self.inputfiles = (inputfiles,)
        self.refseq = refseq
        self.labeltype = kwargs.get("labeltype", "none")
        if self.labeltype is None:
            self.labeltype = "none"
        self.paramfile = kwargs.get("paramfile", Predefined_gpG.default.value)
        self.sampleruncontainer = sampleruncontainer #depreciated
        self.sampleruncontainers = sampleruncontainers
        self.geneignore = kwargs.get("geneignore", Predefined_gpG.geneignore.value)
        # self.record_no = kwargs.get("record_no", "none")
        # self.run_no = kwargs.get("run_no", "none")
        # self.search_no = kwargs.get("search_no", "none")

    def create(self, sampleruncontainers=None, **kwargs):
        if "force" in kwargs:
            force = kwargs.pop("force")
        else:
            force = self.force
        res = super().create(sampleruncontainers=sampleruncontainers, **kwargs)
        if sampleruncontainers is None:
            sampleruncontainers = self.sampleruncontainers

        logger.info(f"gpGrouper {self} create")
        if sampleruncontainers is None:
            logger.error("No sampleruncontainers passed")
            # `raise ValueError(f"Must pass an iterable of SampleRunContainers")

        # set([x.rec_run_search for x in sampleruncontainers])

        _done = set()

        for ix, samplerun_container in enumerate(sampleruncontainers):
            if samplerun_container.psms_filePath is None:
                logger.info(f"skipping {samplerun_container} because no psms file")
                continue
            if (
                samplerun_container.get_file("e2g_QUAL") is not None
                and bool(force) == False
            ):
                logger.info(
                    f"Found e2g_QUAL file for {samplerun_container}, not regrouping"
                )
                continue
            # if samplerun_container.rec_run_search == "52042_1_9":
            #    import ipdb

            #    ipdb.set_trace()
            #    1 + 1
            if samplerun_container.rec_run_search in _done:  # keeps track above
                continue
            kws = self.__dict__.copy()
            # kws["inputfiles"] = samplerun_container
            kws["sampleruncontainer"] = samplerun_container
            kws["name"] = f"{self}-{ix}"
            kws["receiver"] = kws["_receiver"]
            # set_trace()
            yield gpGrouper(**kws)

            _done.add(samplerun_container.rec_run_search)

    @property
    def CMD(self):
        "property that returns psms file info from a `SampleRunContainer` object for gpGrouper"
        # psms_file = self.inputfiles[0].get_file("for-gpg")
        # print(self.inputfiles[0])
        # print(self.inputfiles[0].stem)
        # print(psms_file)

        if self.sampleruncontainer is None:
            return
        sampleruncontainer = self.sampleruncontainer
        psms_file = sampleruncontainer.psms_filePath
        # import ipdb; ipdb.set_trace()
        if psms_file is None:
            return  # not good

        # samplerun_container.set_recrunsearch()
        # psms_file = samplerun_container.psms_filePath
        record_no = sampleruncontainer.record_no
        run_no = sampleruncontainer.run_no
        search_no = sampleruncontainer.search_no
        # psms_file = samplerun_container.psms_filePath
        _ion_score_bins = [8, 10, 13]
        if search_no == "1":
            _ion_score_bins = [10, 20, 30]

        BASE = [
            "gpgrouper",
            "run",
            #
            "-p",
            psms_file,
            #
            "-d",
            self.refseq,
            #
            "--ion-score-bins",
            *_ion_score_bins,
            #
            "-s",
            self.paramfile,
            #
            "--labeltype",
            self.labeltype,
            #
            "--record-no",
            record_no,
            "--run-no",
            run_no,
            "--search-no",
            search_no,
            "--taxonid",
            #
            "9606",
            #
            "--workers",
            self.workers,
            #
            "--outdir",
            str(psms_file.parent),
            "-c",
            self.geneignore
        ]
        if self.phospho:
            BASE.append("--phospho")
        if self.no_taxa_redistrib:
            BASE.append("--no-taxa-redistrib")
        return BASE
