import os
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path
from time import time

from mspcrunner.containers import RunContainer

from .commands import Command, Receiver
from .logger import get_logger

logger = get_logger(__name__)

BASEDIR = os.path.split(__file__)[0]


class Rmd(Command):
    """ 
    Rscript -e \
        'library(rmarkdown)
         rmarkdown::render("/path/to/template.Rmd", "out.html")'
    
    """

    NAME = "Rmder"

    def __init__(
        self,
        *args,
        receiver: Receiver,
        template_file="<#^#>",
        inputfiles=tuple(),
        **kwargs,
    ):

        if "receiver" in kwargs:
            receiver = kwargs.pop("receiver")

        super().__init__(*args, receiver=receiver, **kwargs)
        if "paramfile" in kwargs:
            paramfile = kwargs.pop("paramfile")

        # config = self.read_config(paramfile)

    @property
    def CMD(self):

        some_r_script = None
        some_template = None

        cmd = [
            "RScript",
            some_r_script,
            some_template,
        ]
