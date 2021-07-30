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
        self, receiver, inputfiles, paramfile, outdir, name, refseq=None, **kwargs
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
        psms_file = self.inputfiles[0].get_file("for-gpg")
        print(self.inputfiles[0])
        print(self.inputfiles[0].stem)
        print(psms_file)

        if psms_file is None:
            return tuple(
                "",
            )

        recrun = find_rec_run(psms_file)
        if len(recrun) != 2:
            raise ValueError(f"find_rec_run returned {recrun} from {psms_file}")
        record_no, run_no = recrun
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
            6,
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
