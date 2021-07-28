import time
import os
import logging
from pathlib import Path
import zmq
import random
import sys
import time
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler

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

    def __init__(self, *args, IP=IP, PORT=PORT, **kws):
        super().__init__(*args, **kws)
        self.IP = IP
        self.PORT = PORT

    def on_created(self, event):
        logging.info(f"found {event}")
        #if event.is_directory or not event.src_path.endswith(".raw"):
        #    logging.info(f"ignoring {event}")
        #    return
        
        logging.info('conecting socket')

        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        addr = f"tcp://{self.IP}:{self.PORT}"
        print(addr)
        socket.connect(addr)

        logging.info(f'socket {addr}')

        # see if we are connected
        # or not
        # msg = socket.recv()
        # logging.info(msg)

        #print(msg)

        file = Path(event.src_path)

        socket.send_string(file.name)

        socket.send(
            file.read_bytes()
        )

        print("done sending")

        socket.close()

def test():
    while True:

        msg = socket.recv()
        print(msg)
        # socket.send(b"client message to server1")
        # socket.send(b"client message to server2")
        f = '/mnt/f/45873_1_6/45873_1_MSPCL_559_TMT16_prof_1ug_F24.raw'
        #socket.send(b'test')

        #socket.send_json(f, zmq.SNDMORE)
        logging.info("Sending metadata")
        socket.send_string(os.path.split(f)[-1], zmq.SNDMORE)

        logging.info("reading file")
        raw = open(f, 'rb')
        logging.info("sending file")
        socket.send(raw.read())

    time.sleep(1)

def watch(path='.', IP=IP, PORT=PORT):

    logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    #path = sys.argv[1] if len(sys.argv) > 1 else '.'
    #event_handler = LoggingEventHandler()
    event_handler = NewFile(IP=IP, PORT=PORT)
    observer = Observer()
    observer.schedule(event_handler, str(path), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    #event_handler = LoggingEventHandler()
    event_handler = NewFile()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()