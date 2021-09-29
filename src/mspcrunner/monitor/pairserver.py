import os
import logging
from multiprocessing import Process

try:
    import zmq
except ImportError:
    pass
import random
import sys
import time

# IP = "10.16.1.24"
IP = "*"
PORT = "5556"


def run_server(IP=IP, PORT=PORT):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)

    addr = f"tcp://{IP}:{PORT}"
    print(f"Listening on {addr}", flush=True)

    socket.bind(addr)

    while True:
        #  Wait for next request from client

        # socket.send_string('I am ready')
        flags = 0
        # md = socket.recv_json(flags=flags)

        # first get file name, then get data
        md = socket.recv_string(flags=0)
        msg = socket.recv()

        print(md)
        # print(msg)
        print("getting ready to write")
        if os.path.exists(md):
            print(f"Already have {md}")
            # socket.send_string(f'Already have {md}')
            continue

        with open(md, "wb") as out:
            print("writing")
            out.write(msg)
        print("done")

        # print(f"Received request: {message}")

        #  Do some 'work'
        # time.sleep(1)
