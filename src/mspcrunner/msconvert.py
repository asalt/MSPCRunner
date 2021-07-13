# MSCONVERT

from collections import OrderedDict
import sys
import os
from time import time
from pathlib import Path
from .logger import get_logger
from .commands import Command

logger = get_logger(__name__)

#from .config import get_masic_exe

_EXE = None


def get_exe():
    global _EXE
    if _EXE is None:
        return get_masic_exe()
    return _EXE


# BASEDIR = os.path.split(__file__)[0]
# # MSFRAGGER_EXE = os.path.join(BASEDIR, "..\ext\MSFragger\MSFragger-3.2\MSFragger-3.2.jar")
# MASIC_EXE = os.path.abspath(os.path.join(BASEDIR, "../../ext/MASIC/MASIC_Console.exe"))
# MASIC_EXE = Path(MASIC_EXE)


class MSConvert(Command):

    NAME = "MSConvert"

    @property
    def CMD(self):
        # TODO check os
        # if not self.inputfiles:
        #    return

        self.announce()

        inputfile = self.inputfile.get_file("spectra")

        # spectra_files = list()
        # # if self.inputfiles:
        # for filecontainer in self.inputfiles:
        #     calculated_result_name = filecontainer.get_file("SICs")
        #     if calculated_result_name is not None:
        #         logger.info(f"Already detect {str(calculated_result_name)}")
        #         continue  # already processed
        #     spectra_files.append(filecontainer.get_file("spectra"))

        # spectra_files = filter(None, spectra_files)
        # if len(spectra_files) > 1:
        #     raise ValueError("")
        # if len(spectra_files) == 1:
        #     inputfile = spectra_files[0]

        # yield [
        #     "mono",
        #     MASIC_EXE,
        #     f"/P:{self.paramfile}",
        #     f"/O:{os.path.dirname(inputfile)}",
        #     *[f"/I:{inputfile}" for inputfile in spectrafiles],
        #     f"/O:{inputfile.parent.resolve()}",
        #     f"/I:{inputfile.resolve()}",
        # ]
        # for inputfile in spectra_files:
        #MASIC_EXE = get_exe()

        return [ 'docker', 'run', '-it', '--rm',  '-v', "($pwd):/data",
         "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses",
          "wine", "msconvert",
          inputfile
        ]