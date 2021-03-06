import typer
import logging
from pathlib import Path

from . import pairserver, pairclient
from .qc import watch as mspcrunner_watch

app = typer.Typer(name="monitor", result_callback=None)

_DEFAULT_IP = "10.16.1.24"
# _DEFAULT_IP = "*"


@app.command("receive")
def receive(
    PORT: int = 5556,
    IP: str = "*",
):
    logging.info("Starting server")
    pairserver.run_server(IP=IP, PORT=PORT)


@app.command("watch")
def watch(
    path: Path = ".",
    PORT: int = 5556,
    IP: str = _DEFAULT_IP,
    ext: str = "raw",
):
    pairclient.watch(path=path, IP=IP, PORT=PORT, ext=ext)


@app.command("qc")
def qc(path: Path = "."):
    """
    monitor `path` for new QC files and process them
    """
    mspcrunner_watch()
