import sys
import logging

from time import time


def get_logger(name=__name__):

    # import queue
    # from logging import handlers
    # que = queue.Queue(-1)  # no limit on size
    # queue_handler = handlers.QueueHandler(que)
    # handler = logging.StreamHandler()
    # listener = handlers.QueueListener(que, handler)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # logger.addHandler(queue_handler)

    fh = logging.FileHandler("MSPCRunner.log")
    # fh.flush = sys.stdout.flush
    # fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    # ch.flush = sys.stdout.flush
    # ch = logging.StreamHandler()

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # listener.start()
    # logging.getLogger('').addHandler(fh)
    return logger
