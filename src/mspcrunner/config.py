import sys
from configparser import ConfigParser
from pathlib import Path
import click
import typer

APPDIR = Path(typer.get_app_dir("mspcrunner"))
APPCONF = APPDIR / "mspcrunner.conf"

# @app.callback(
#    invoke_without_command=True, no_args_is_help=True, result_callback=worker_run
# )


def get_current_context():
    """
    can make our own context
    """

    ctx = click.get_current_context(silent=True)
    # if ctx is None:
    #     ctx = make_new_context()

    return ctx


def graceful_exit():
    ...
    # ctx = get_current_context()
    # sys.exit(0)
    # ctx = click.get_current_context(silent=False)
    # ctx.exit(0)


# TODO figure out how to exit program after config is called
config_app = typer.Typer(name="config", result_callback=None)


_CONFIG = None


def get_conf():
    """get_conf return MSPCRunner ConfigParser object
    read from file if exists
    else make a new one
    """
    global _CONFIG

    if _CONFIG is None:
        _CONFIG = ConfigParser()

    if isinstance(_CONFIG, ConfigParser):
        return _CONFIG
    return load_config()


def load_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = ConfigParser()

    if APPCONF.exists():
        return _CONFIG.read_file(APPCONF)

    _CONFIG["paths"] = {
        "basepath": "",
        "msfragger-path": "",
        "masic-path": "",
        "msfragger-executable": "",
        "masic-executable": "",
    }
    return _CONFIG


def write_config():
    conf = get_conf()
    conf.write(APPCONF)


@config_app.command("set-dir")
def set_dir(dir: Path):
    """set_dir directory that contains subdirectories for MASIC, MSFragger, etc

    Update mspcrunner configuration
    """

    conf = get_conf()

    typer.echo("set all")


@config_app.command("set-msfragger-dir")
def set_msfragger_dir():
    typer.echo("msfragger dir")


@config_app.command("set-masic-dir")
def set_masic_dir():
    typer.echo("masic dir")


def get_msfragger_exe():
    return (
        Path.home() / "Documents/MSPCRunner/MSFragger/MSFragger-3.2/MSFragger-3.2.jar"
    ).resolve()


def get_masic_exe():
    return (Path.home() / "Documents/MASIC/MASIC_Console.exe").resolve()
