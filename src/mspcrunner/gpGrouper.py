# gpGrouper.py


import logging
import os

from ipdb.__main__ import set_trace

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
        # samplerun_containers,
        paramfile=None,
        inputfiles=None,
        outdir=None,
        name=None,
        refseq=None,
        **kwargs
    ):

        super().__init__(
            receiver,
            paramfile=paramfile,
            inputfiles=inputfiles,
            name=name,
            refseq=refseq,
            **kwargs
        )
        # super().__init__(
        #     receiver,
        #     # inputfiles=inputfiles,
        #     paramfile=paramfile,
        #     outdir=outdir,
        #     name=name,
        #     **kwargs,
        # )
        self.phospho = kwargs.get("phospho", False)
        # self.inputfiles = (inputfiles,)
        self.refseq = refseq
        self.labeltype = kwargs.get("labeltype", "none")
        if self.labeltype is None:
            self.labeltype = "none"
        self.paramfile = kwargs.get("paramfile", Predefined_gpG.default.value)
        # self.record_no = kwargs.get("record_no", "none")
        # self.run_no = kwargs.get("run_no", "none")
        # self.search_no = kwargs.get("search_no", "none")

    @property
    def CMD(self):
        "property that returns psms file info from a `SampleRunContainer` object for gpGrouper"
        # psms_file = self.inputfiles[0].get_file("for-gpg")
        # print(self.inputfiles[0])
        # print(self.inputfiles[0].stem)
        # print(psms_file)

        if self.inputfiles is None:  # this is not supposed to happen
            return

        for samplerun_container in self.inputfiles:
            # import ipdb ipdb.set_trace()

            samplerun_container.set_recrunsearch()
            # psms_file = samplerun_container.psms_filePath
            psms_file = samplerun_container.psms_file
            record_no = samplerun_container.record_no
            run_no = samplerun_container.run_no
            search_no = samplerun_container.search_no
            # psms_file = samplerun_container.psms_filePath

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
                8,
                #
                "--outdir",
                str(psms_file.parent),
            ]
            if self.phospho:
                BASE.append("--phospho")
            return BASE
