import os
import sys
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path
from time import time
import ipdb

from mspcrunner.containers import RunContainer

from .commands import MSFRAGGER_EXE, Command, Receiver
from .logger import get_logger

logger = get_logger(__name__)

BASEDIR = os.path.split(__file__)[0]

# MSFRAGGER_EXE = os.path.abspath(
#     os.path.join(BASEDIR, "../../ext/MSFragger/MSFragger-3.2/MSFragger-3.2.jar")
# )
# MSFRAGGER_EXE = Path(MSFRAGGER_EXE)


from .config import get_msfragger_exe, write_config

_MSFRAGGER_EXE = None


def get_exe():
    global _MSFRAGGER_EXE
    if _MSFRAGGER_EXE is None:
        _MSFRAGGER_EXE = get_msfragger_exe()
        return _MSFRAGGER_EXE
        # print(x, x is None)
        # if x is None:
        #     raise ValueError("MSFRAGGER EXE does not exist")
        # return x
    return _MSFRAGGER_EXE


class MSFragger(Command):

    NAME = "MSFragger"

    def __init__(
        self,
        *args,
        receiver: Receiver,
        inputfiles=tuple(),
        ramalloc="50G",
        refseq=None,
        paramfile=None,
        **kwargs,
    ):

        if "receiver" in kwargs:
            receiver = kwargs.pop("receiver")

        super().__init__(*args, receiver=receiver, **kwargs)
        self.ramalloc = ramalloc
        self._config = None
        self.inputfiles = inputfiles
        # self.runcontainers = runcontainers
        self.refseq = refseq
        self.paramfile = paramfile
        self.local_paramfile = None

        if paramfile is not None:
            config = self.read_config(paramfile)
            self._config = config
            self.local_paramfile = paramfile.name

        # we can edit the parameters right here
        self.set_param("database_name", str(refseq))
        self.set_param("data_type", "0")
        # if refseq is not None:
        #     config["top"]["database_name"] = str(refseq)

    def write_config(self):
        # should be called `write_config_and_set_self.paramfile_to_local_copy`

        config_out = self.local_paramfile
        with open(config_out, "w") as fh:
            logger.info(f"Writing {config_out}")
            self._config.write(fh)

            fh.seek(0)
            fh.write("\b" * len("[top]"))
        self.paramfile = Path(config_out)
        return self.paramfile

    def read_config(
        self, conf
    ) -> ConfigParser:  # this is smarter than using csv module

        parser = ConfigParser(inline_comment_prefixes="#")
        parser.optionxform = str  # preserve case

        with open(conf) as stream:
            # print(stream.read().strip(r"\x08"))
            string = stream.read().strip(
                "\x08"
            )  # have to strip these bytes that sometimes exist
            parser.read_string("[top]\n" + string)  # create dummy header
        return parser

    def create(self, runcontainers=None, **kwargs):
        if runcontainers is None:
            yield self
        else:
            d = self.__dict__.copy()
            for kw in kwargs:
                if kw in d:
                    d.update(kw, kwargs[kw])
                # d["inputfiles"] = container  # depreciate
                # d["container"] = container  # depreciate
                ix=0
                d["name"] = d.get("name", "name") + f"-{ix}"
            if "receiver" not in d and "_receiver" in d:
                d["receiver"] = d["_receiver"]
            yield type(self)(**d)


    # @property
    # def params() -> dict:
    #    if self._params is None:
    #        param_dict = read_params(self.conf)
    #        self._params = param_dict
    #    return self._params

    # def get_param_value(self, param):

    #    return self._params.get(param)

    def set_param(self, param, value):
        # if param not in self._config:
        #     logger.error(f"{param} does not exist")
        if self._config is None:
            logger.warning(f"No param file set for {self}")
            return -1
        self._config["top"][param] = str(value)
        return 0

    @property
    def CMD(self):
        # spectra_files = "<not set>"

        if self.paramfile is not None:
            self.write_config()

        spectra_files = list()
        MSFRAGGER_EXE = get_exe()

        # inputfile = self.inputfiles[0].get_file("spectra")

        if not self.inputfiles:
            return
            # raise ValueError("then why are we here?")

        for run_container in self.inputfiles:

            spectraf = run_container.get_file("spectra")
            search_res = run_container.get_file("tsv_searchres")
            if search_res is not None:
                logger.info(f"{search_res} exists for {run_container}")
                continue

            if spectraf is not None:
                spectra_files.append(spectraf)
            else:
                logger.info(f"cannot find spectra file {run_container}")

        if len(spectra_files) == 0:
            logger.debug(f"no new files found for search")
            return

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
