import time
import os
import logging
from pathlib import Path
import random
import sys
import time
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler

# watchdog newfile send
class NewQCFile(FileSystemEventHandler):
    def __init__(self, *args, search_setting="LF-OTIT", refseq="hs2020", **kws):
        super().__init__(*args, **kws)
        self.search_setting = search_setting
        self.refseq = refseq

    def on_created(self, event):
        logging.info(f"found {event}")

        if event.is_directory:
            logging.info(f"ignoring {event}")
            return
        if self.ext and not event.src_path.endswith(self.ext):
            logging.info(f"ignoring {event}")
            return

        logging.info("file")

        CMD = [
            "mspcrunner",
            "run",
            "--path",
            ".",  # TODO make more specific
            "search",
            "--preset",
            "LF-OTIT",
            "--refseq",
            "hs2020"
        ]
