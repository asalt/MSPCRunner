import ipdb
from ipdb import set_trace
from collections import OrderedDict, defaultdict
from pathlib import Path
import sys
from time import time
import typing

from mspcrunner.containers import RunContainer, SampleRunContainer
from .logger import get_logger
import ipdb

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
        self._runcontainers = defaultdict()
        self._sampleruncontainers = defaultdict()

    # def __new__(self):
    #     self._runcontainers = list()
    #     return self

    def add_sampleruncontainer(self, container):
        if isinstance(container, SampleRunContainer):
            # self._runcontainers.append(container)
            self._sampleruncontainers[hash(container)] = container
        else:
            raise ValueError(f"{container} must be of type {SampleRunContainer}")

    def add_runcontainer(self, container):
        if isinstance(container, RunContainer):
            self._runcontainers[hash(container)] = container
        else:
            raise ValueError(f"{container} must be of type {RunContainer}")

    def add_container(self, container):
        if isinstance(container, SampleRunContainer):
            self.add_sampleruncontainer(container)
        elif isinstance(container, RunContainer):
            self.add_runcontainer(container)
        else:
            raise ValueError(f"{container} must be of type {SampleRunContainer}")

    def get_runcontainers(self):
        return [x for x in self._runcontainers.values()]

    def get_sampleruncontainers(self):
        return [x for x in self._sampleruncontainers.values()]

    def register(self, command_name, command):
        self._commands[command_name] = command

    def update_inputfiles(self, cmd):
        """
        do not use this right now
        """
        cmd.update_inputfiles(*[x for x in self.get_runcontainers().values()])
        cmd.update_inputfiles(*[x for x in self.get_sampleruncontainers().values()])

    def execute(self, command_name):
        "Execute any registered commands"

        # if 'experiment_finder' in self._commands and 'experiment_finder' in self._output:
        # the_files = self._output.get("experiment_finder", None)

        if command_name in self._commands.keys():

            # the_command = self._commands[command_name]
            # # the_command.set_files(the_files)

            print(command_name)
            cmd = self._commands[command_name]

            if isinstance(cmd, tuple()):
                pass
            # if command_name.startswith("build"):

            factory = cmd.create(
                runcontainers=self.get_runcontainers() or None,
                sampleruncontainers=self.get_sampleruncontainers() or None,
            )
            if command_name == "Rmd":
                import ipdb

                ipdb.set_trace()
            if (
                command_name.startswith("concat")
                or command_name == "psms-collector-for-concat"
                or command_name == "SampleRunContainerBuilder"
                or command_name == "runbuilder"
            ):
                pass

            # ipdb.set_trace()
            for obj in factory:

                self._history.append((time(), command_name))

                # set_trace()
                output = obj.execute()
                # print(obj)
                # print(output)
                self._output[command_name] = output

                # input files not dynamically found, but should be
                # self.update_inputfiles(cmd)

                # cmd.set_files(the_files)

                if output is not None and isinstance(output, typing.Iterable):

                    for o in output:
                        if isinstance(o, RunContainer) and o.n_files != 0:
                            self.add_runcontainer(o)
                        elif isinstance(
                            o, SampleRunContainer
                        ):  # check if SampleRunContainer first, then count
                            # number of associated RunContainers
                            if len(o.runcontainers) != 0:
                                self.add_sampleruncontainer(o)
                            # elif isinstance(o, SampleRunContainer)
                            else:
                                # import ipdb; ipdb.set_trace()
                                pass

            # return output  # return nothing/...
            return  # return nothing/...

        else:
            logger.error(f"Command [{command_name}] not recognized")

    def get_files(self):
        """get_files [depreciated]


        Returns:
            [type]: [description]
        """
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
