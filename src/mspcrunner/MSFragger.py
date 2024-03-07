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
        # inputfiles=tuple(),
        ramalloc="50G",
        refseq=None,
        paramfile=None,
        force=False,
        containers=None, # depreciate
        runcontainers=None,
        **kwargs,
    ):
        if "receiver" in kwargs:
            receiver = kwargs.pop("receiver")

        super().__init__(
            *args, receiver=receiver,
            containers=containers,  # depreciate
            runcontainers=runcontainers,
            force=force, **kwargs
        )
        self.ramalloc = ramalloc
        self.runcontainers = runcontainers
        self._config = None
        # self.runcontainers = runcontainers
        self.refseq = refseq
        if paramfile is None:
            self.paramfile = paramfile
        if paramfile is not None:
            self.paramfile = Path(paramfile)
            config = self.read_config(paramfile)
            self._config = config

        # we can edit the parameters right here
        self.set_param("database_name", str(refseq))
        self.set_param("data_type", "0")
        self.set_param("allow_multiple_variable_mods_per_residue", "1")
        # self.set_param("check_", "0")
        self.set_param("minimum_ratio", "0.01")
        self.set_param("output_format", "tsv_pepxml_pin")
        # if refseq is not None:
        #     config["top"]["database_name"] = str(refseq)

    def write_config(self):
        # should be called `write_config_and_set_self.paramfile_to_local_copy`

        # config_out = self.local_paramfile
        config_out = Path(".") / f"{self.paramfile.stem}_mspcrunner.params"
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
        import ipdb; ipdb.set_trace()
        if self.paramfile is not None:
            self.write_config()
        d = self.__dict__.copy()
        # d['name'] = d.get("name", "name") + f"-{len(runcontainers)}" # this is a bit of a hack
        d['runcontainers'] = runcontainers
        yield type(self)(**d) # I think this solves the problem of spawning multiple objects with the same files
        # for _ in range(1): # return just 1 object with all runcontainers to be searched all together
        #     yield type(self)(**d) # I think this solves the problem of spawning multiple objects with the same files
        # return [self]

        #return super().create(runcontainers=runcontainers, **kwargs)
    # def create(self, runcontainers=None, **kwargs):
    #     if runcontainers is None:
    #         yield self
    #     else:
    #         # from copy import deepcopy

    #         d = self.__dict__.copy()

    #         for kw in kwargs:
    #             if kw in d:
    #                 d.update(kw, kwargs[kw])
    #             # d["inputfiles"] = container  # depreciate
    #             # d["container"] = container  # depreciate
    #             ix = 0
    #             d["name"] = d.get("name", "name") + f"-{ix}"
    #         # finally add the containers
    #         d["containers"] = runcontainers
    #         if "receiver" not in d and "_receiver" in d:
    #             d["receiver"] = d["_receiver"]
    #         yield type(self)(**d)

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

    def get_param(self, param):
        if self._config is None:
            logger.warning(f"No param file set for {self}")
            return -1
        return self._config["top"][param]

    @property
    def CMD(self):
        # spectra_files = "<not set>"


        # import ipdb; ipdb.set_trace()
        spectra_files = list()
        MSFRAGGER_EXE = get_exe()

        # inputfile = self.inputfiles[0].get_file("spectra")

        if not self.runcontainers:
            return
            # raise ValueError("then why are we here?")

        for run_container in self.runcontainers:
            spectraf = run_container.get_file("spectra")
            if spectraf is None:
                logger.info(f"cannot find spectra file {run_container}, skipping")
                continue
            search_res = run_container.get_file("tsv_searchres")
            if search_res is not None and not self.force:
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
