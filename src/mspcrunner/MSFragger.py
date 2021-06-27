from collections import OrderedDict
import os
import sys
from pathlib import Path
from time import time
from configparser import ConfigParser

from .logger import get_logger
from .commands import Command

logger = get_logger(__name__)

BASEDIR = os.path.split(__file__)[0]

MSFRAGGER_EXE = os.path.abspath(
    os.path.join(BASEDIR, "../../ext/MSFragger/MSFragger-3.2/MSFragger-3.2.jar")
)
MSFRAGGER_EXE = Path(MSFRAGGER_EXE)


class MSFragger(Command):

    NAME = "MSFragger"

    def __init__(
        self, *args, inputfiles=tuple(), ramalloc="50G", refseq=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if "paramfile" in kwargs:
            paramfile = kwargs.pop("paramfile")
        self.ramalloc = ramalloc
        self._params = None
        self.inputfiles = inputfiles
        self.refseq = refseq
        self.configfile = None

        config = self.read_config(paramfile)

        # todo put in separate methods
        if refseq is not None:
            config["top"]["database_name"] = refseq

        config_out = f"{paramfile.name}"
        with open(config_out, "w") as fh:
            logger.info(f"Writing {config_out}")
            config.write(fh)

            fh.seek(0)
            fh.write("\b" * len("[top]"))
        self.paramfile = Path(config_out)

        # write_config
        1 + 1

    def read_config(
        self, conf
    ) -> ConfigParser:  # this is smarter than using csv module

        parser = ConfigParser(inline_comment_prefixes="#")
        parser.optionxform = str  # preserve case

        with open(conf) as stream:
            parser.read_string("[top]\n" + stream.read())  # create dummy header
        return parser

    # @property
    # def params() -> dict:
    #    if self._params is None:
    #        param_dict = read_params(self.conf)
    #        self._params = param_dict
    #    return self._params

    # def get_param_value(self, param):

    #    return self._params.get(param)

    def set_param(self, param, value):
        if param not in self._params:
            logger.error(f"{param} does not exist")
        self._params[param] = value

    @staticmethod
    def quote_if_windows(x):  # not needed
        if platform.system() == "Windows":
            return f'"{x}"'
        return f"{x}"

    @property
    def CMD(self):
        # spectra_files = "<not set>"
        spectra_files = tuple()
        if self.inputfiles:
            spectra_files = [
                x.get_file("spectra") for x in self.inputfiles
            ]  # the values are RawFile instances
            spectra_files = [str(x.resolve()) for x in spectra_files if x is not None]
        return [
            "java",
            f"-Xmx{self.ramalloc}G",
            "-jar",
            # self.quote_if_windows(MSFRAGGER_EXE),
            # self.quote_if_windows(self.paramfile.resolve()),
            # f"\"{MSFRAGGER_EXE}\"",
            f"{MSFRAGGER_EXE.resolve()}",
            f"{self.paramfile.resolve()}",
            *spectra_files
            # *self.get_inputfiles(),
        ]