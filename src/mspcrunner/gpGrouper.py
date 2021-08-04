# gpGrouper.py


import logging
import os

from .commands import Command
from .utils import find_rec_run
from .predefined_params import Predefined_gpG

BASE = [
    "gpgrouper",
    "run",
    "--ion-score-bins",
    "16 18 20",
    "-s",
    "./params/gpgrouper.conf",
    "--taxonid",
    "9606",
    "--workers",
    8,
]


class gpGrouper(Command):

    NAME = "gpGrouper"

    def __init__(
        self,
        receiver,
        samplerun_container,
        paramfile,
        outdir,
        name,
        refseq=None,
        **kwargs
    ):
        super().__init__(
            receiver,
            inputfiles=inputfiles,
            paramfile=paramfile,
            outdir=outdir,
            name=name,
            **kwargs,
        )
        self.phospho = kwargs.get("phospho", "False")
        self.inputfiles = (inputfiles,)
        self.refseq = refseq
        self.labeltype = kwargs.get("labeltype", "none")
        self.paramfile = kwargs.get("paramfile", Predefined_gpG.default.value)
        self.record_no = kwargs.get("record_no", "none")
        self.run_no = kwargs.get("run_no", "none")
        self.search_no = kwargs.get("search_no", "none")

    @property
    def CMD(self):
        "property that returns psms file info from a `SampleRunContainer` object for gpGrouper"
        # psms_file = self.inputfiles[0].get_file("for-gpg")
        # print(self.inputfiles[0])
        # print(self.inputfiles[0].stem)
        # print(psms_file)

        psms_file = self.samplerun_container.psms_filePath
        import ipdb

        ipdb.set_trace()

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
            "16",
            "18",
            "20",
            #
            "-s",
            self.paramfile,
            #
            "--labeltype",
            self.labeltype,
            #
            "--record-no",
            self.record_no,
            "--run-no",
            self.run_no,
            "--search-no",
            self.search_no,
            "--taxonid",
            #
            "9606",
            #
            "--workers",
            8,
            #
            "--outdir",
            str(psms_file.parent),
        ]
        if self.phospho:
            BASE.append("--phospho")
        return BASE
