import time
import os
import logging
import subprocess
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
        # if self.ext and not event.src_path.endswith(self.ext):
        #     logging.info(f"ignoring {event}")
        #     return

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
            "hs2020",
        ]
        subprocess.run(CMD)


def watch(path="."):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # path = sys.argv[1] if len(sys.argv) > 1 else '.'
    # event_handler = LoggingEventHandler()
    event_handler = NewQCFile()
    observer = Observer()
    observer.schedule(event_handler, str(path), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
