import os
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path
from time import time

from mspcrunner.containers import RunContainer

from .commands import Command, Receiver
from .predefined_params import PREDEFINED_RMD_TEMPLATES
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
        template_file: Path = None,
        # inputfiles=tuple(),
        **kwargs,
    ):

        self.template_file = template_file

        if "receiver" in kwargs:
            receiver = kwargs.pop("receiver")

        super().__init__(*args, receiver=receiver, **kwargs)
        if "paramfile" in kwargs:
            paramfile = kwargs.pop("paramfile")

        # config = self.read_config(paramfile)

    @property
    def CMD(self):
        """
        render                package:rmarkdown                R Documentation

        Render R Markdown

        Description:

            Render the input file to the specified output format using pandoc.
            If the input requires knitting then ‘knit’ is called prior to
            pandoc.

        Usage:

            render(
            input,
            output_format = NULL,
            output_file = NULL,
            output_dir = NULL,
            output_options = NULL,
            output_yaml = NULL,
            intermediates_dir = NULL,
            knit_root_dir = NULL,
            runtime = c("auto", "static", "shiny", "shiny_prerendered"),
            clean = TRUE,
            params = NULL,
            knit_meta = NULL,
            envir = parent.frame(),
            run_pandoc = TRUE,
            quiet = FALSE,
            encoding = "UTF-8"
        )

        params: A list of named parameters that override custom params
                specified within the YAML front-matter (e.g. specifying a
                dataset to read or a date range to confine output to). Pass
                ‘"ask"’ to start an application that helps guide parameter
                configuration.




        """

        some_r_script = None
        some_template = None
        # template_file = self.template_file.absolute()
        template_file = self.template_file.absolute()
        outname = ".html"
        title = "title"
        directory = "."

        import ipdb

        # ipdb.set_trace()
        params = f"""
            c('title' = {title},
              'directory' = {directory},
              'expids' = 

        """
        output_file = "test.html"
        output_dir = "."

        if self.container is not None:
            import ipdb

            ipdb.set_trace()
        cmd = [
            f"Rscript",
            "-e",
            f"""library(rmarkdown)
            rmarkdown::render("{template_file}", 
            #output_format="html",
            output_file="{output_file}",
            output_dir="{output_dir}",
            params={params}
            )""",
        ]

        # some_template,
        # some_r_script,

        return cmd
