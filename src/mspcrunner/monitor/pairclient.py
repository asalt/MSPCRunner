import time
import os
import logging
from pathlib import Path

_ZMQ = True
try:
    import zmq
except ImportError:
    _ZMQ = False

import random
import sys
import time

_WATCHDOG = True
try:
    from watchdog.observers import Observer
    from watchdog.events import LoggingEventHandler, FileSystemEventHandler
except ImportError:
    _WATCHDOG = False
    FileSystemEventHandler = object

# ZMQ setup
IP = "10.16.1.24"
IP = "localhost"
PORT = "5556"
# context = zmq.Context()
# socket = context.socket(zmq.PAIR)
# #socket.connect("tcp://localhost:%s" % port)
# socket.connect(f"tcp://{IP}:{PORT}")
# ==================================

# watchdog newfile send
class NewFile(FileSystemEventHandler):
    def __init__(self, *args, IP=IP, PORT=PORT, ext=None, **kws):
        super().__init__(*args, **kws)
        self.IP = IP
        self.PORT = PORT
        self.ext = ext

    def on_created(self, event):
        logging.info(f"found {event}")

        if event.is_directory:
            logging.info(f"ignoring {event}")
            return
        if self.ext and not event.src_path.endswith(self.ext):
            logging.info(f"ignoring {event}")
            return

        logging.info("conecting socket")

        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        addr = f"tcp://{self.IP}:{self.PORT}"
        print(addr)
        socket.connect(addr)

        logging.info(f"socket {addr}")

        # see if we are connected
        # or not
        # msg = socket.recv()
        # logging.info(msg)
        # print(msg)

        file = Path(event.src_path)

        # socket.send_string(file.name)
        # socket.send(file.read_bytes())

        socket.send_multipart(bytes(file.name), 'utf8')
        socket.send_multipart(bytes(file.name), 'utf8')


        print("done sending")

        socket.close()


def watch(path=".", IP=IP, PORT=PORT, ext=None):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # path = sys.argv[1] if len(sys.argv) > 1 else '.'
    # event_handler = LoggingEventHandler()
    event_handler = NewFile(IP=IP, PORT=PORT, ext=ext)
    observer = Observer()
    observer.schedule(event_handler, str(path), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    # event_handler = LoggingEventHandler()
    event_handler = NewFile()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
