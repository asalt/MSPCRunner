import platform
from collections import OrderedDict
import sys
import os
from time import time
from pathlib import Path
from .logger import get_logger
from .commands import Command

logger = get_logger(__name__)

from .config import get_masic_exe

_EXE = None


def get_exe():
    global _EXE
    if _EXE is None:
        _EXE = get_masic_exe()
    return _EXE


# BASEDIR = os.path.split(__file__)[0]
# # MSFRAGGER_EXE = os.path.join(BASEDIR, "..\ext\MSFragger\MSFragger-3.2\MSFragger-3.2.jar")
# MASIC_EXE = os.path.abspath(os.path.join(BASEDIR, "../../ext/MASIC/MASIC_Console.exe"))
# MASIC_EXE = Path(MASIC_EXE)


class MASIC(Command):

    NAME = "MASIC"

    # def __init__(self, receiver, containers, paramfile, outdir, name, **kwargs):
    def __init__(self, receiver, paramfile, **kwargs):
        super().__init__(
            receiver,
            # containers=containers,
            # paramfile=paramfile,
            # outdir=outdir,
            # name=name,
            **kwargs,
        )
        self.paramfile = paramfile
        self._params = None
        self.param_out = None

        param_out = f"{paramfile.name}"
        self.save_params(param_out)

    @property
    def params(self):
        if self._params is None:
            import xml.etree.ElementTree as ET

            tree = ET.parse(self.paramfile)
            self._params = tree
        return self._params

    def create(self, *args, **kwargs):
        import ipdb; ipdb.set_trace()
        res = super().create(*args, **kwargs)
        return res

    def set_attr(self, section_name, item_key, item_value):
        """
        <section name="MasicExportOptions">
            <item key="ReporterIonMassMode", value="0">
        <section name="SICOptions">
            <item key=x>
        """
        self.params
        root = self.params.getroot()
        for section in root.find("section"):
            _name = section.attrib.get("name")
            if _name != section_name:
                continue
            for item in section.find("item"):
                if not item.attrib.get("key") == item_key:
                    continue
                item.set(item_key, item_value)

    def save_params(self, param_out):
        self.params.write(param_out)

    @property
    def CMD(self):

        # TODO check os
        # if not self.inputfiles:
        #    return

        self.announce()

        inputfile = self.inputfile.get_file("raw")
        if inputfile is None:
            inputfile = self.inputfile.get_file("spectra")
        if inputfile is None:  # if still none
            return

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

        MASIC_EXE = get_exe()
        return filter(
            None,
            [
                "mono" if platform.system() == "Linux" else "",
                MASIC_EXE,
                f"/P:{self.paramfile}",
                f"/O:{os.path.dirname(inputfile)}",
                f"/I:{inputfile}",
                # f"/O:{inputfile.parent.resolve()}",
                # f"/I:{inputfile.resolve()}",
            ],
        )
        # return [MASIC_EXE, f"/P:{self.paramfile}", f"/O:{self.outdir}", f"/I:{inputfile}"]
        # return [MASIC_EXE, f"/P:\"{self.paramfile}\"", f"/O:{self.outdir}", f"/I:{inputfile}"]


class MASIC_Tester(MASIC):

    NAME = "MASIC Tester"

    def execute(self):

        print(f"{self!r} : sending {self.CMD}")
        # ret = self._receiver.action("run", self.CMD)
        ret = self._receiver.run(self.CMD)
        # print(f"{self} : sending {self.CMD}")
