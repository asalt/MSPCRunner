from collections import OrderedDict
import sys
from time import time
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

    def something_else(self):
        pass

    def set_on_start(self, command):
        pass
