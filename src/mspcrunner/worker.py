from collections import OrderedDict
from pathlib import Path
import sys
from time import time

from mspcrunner.containers import RunContainer, SampleRunContainer
from .logger import get_logger

logger = get_logger(__name__)


class Worker:  # invoker
    """
    Invoker
    """

    def __init__(
        self,
    ):  # Override init to initialize an Invoker with Commands to accept.
        self._history = list()
        self._commands = dict()
        self._output = OrderedDict()

        self._file = None
        self._path = None
        self._runcontainers = set()
        self._sampleruncontainers = set()

    # def __new__(self):
    #     self._runcontainers = list()
    #     return self

    def add_sampleruncontainer(self, container):
        if isinstance(container, SampleRunContainer):
            # self._runcontainers.append(container)
            self._runcontainers.add(container)
        else:
            raise ValueError(f"{container} must be of type {SampleRunContainer}")

    def add_runcontainer(self, container):
        if isinstance(container, RunContainer):
            self._runcontainers.add(container)
        else:
            raise ValueError(f"{container} must be of type {RunContainer}")

    def add_container(self, container):
        if isinstance(container, SampleRunContainer):
            self.add_sampleruncontainer(container)
        elif isinstance(container, RunContainer):
            self.add_sampleruncontainer(container)
        else:
            raise ValueError(f"{container} must be of type {SampleRunContainer}")

    def get_runcontainers(self):
        return self._runcontainers

    def get_sampleruncontainers(self):
        return self._sampleruncontainers

    def register(self, command_name, command):
        self._commands[command_name] = command

    def execute(self, command_name):
        "Execute any registered commands"

        # if 'experiment_finder' in self._commands and 'experiment_finder' in self._output:
        # the_files = self._output.get("experiment_finder", None)

        if command_name in self._commands.keys():

            # the_command = self._commands[command_name]
            # the_command.set_files(the_files)

            cmd = self._commands[command_name]

            # cmd.set_files(the_files)

            self._history.append((time(), command_name))
            import ipdb

            ipdb.set_trace()
            output = cmd.execute()
            self._output[command_name] = output
            return output

        else:
            logger.error(f"Command [{command_name}] not recognized")

    def get_files(self):
        if (
            "experiment_finder" in self._commands
            and "experiment_finder" not in self._output
        ):
            self.execute("experiment_finder")
        return self._output.get("experiment_finder")

    # def context(self) -> Context:
    #    return self._context

    # @context.setter
    # def context(self, context: Context) -> None:
    #    self._context = context

    @property
    def file(self):
        return self._file

    @property
    def path(self):
        return self._path

    def set_file(self, p: Path):
        self._file = p
        pass

    def set_path(self, p: Path):
        self._path = p
        pass

    def set_on_start(self, command):
        pass
