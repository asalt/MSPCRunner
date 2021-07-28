import typer
import logging
from pathlib import Path

from . import pairserver, pairclient

app = typer.Typer(name='monitor', result_callback=None)

_DEFAULT_IP = "10.16.1.24"
#_DEFAULT_IP = "*"

@app.command('receive')
def receive(
    PORT : int = 5556,
    IP : str = "*"
):
    logging.info("Starting server")
    pairserver.run_server(IP=IP, PORT=PORT)

@app.command('watch')
def watch(
    path : Path =  '.',
    PORT : int = 5556,
    IP : str = _DEFAULT_IP
):
    pairclient.watch(path=path, IP=IP, PORT=PORT)