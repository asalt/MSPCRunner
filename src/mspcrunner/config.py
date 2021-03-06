from configparser import ConfigParser
from genericpath import exists
from pathlib import Path
from _pytest.config import directory_arg
import click

# from mspcrunner import logger
from typing import Container, Optional, List, Tuple
import typer

from .commands import get_logger

logger = get_logger(__name__)

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

SECTIONS = {
    "ext": {
        "basepath": "",
        "msfragger-path": "",
        "masic-path": "",
        "msfragger-executable": "",
        "masic-executable": "",
    },
    "refdb": {
        "hs2020": "",
        "hsmm2020": "",
    },
    # "params": {
    # "masic-params": "",
    # "msfragger-params": "",
    # },
    "search-params": {
        "otot": "",
    },
    "quant-params": {
        "lf": "",
    },
}


def get_conf() -> ConfigParser:
    """get_conf return MSPCRunner ConfigParser object
    read from file if exists
    else make a new one
    """
    global _CONFIG

    if isinstance(_CONFIG, ConfigParser):
        return _CONFIG
    return load_config()


def load_config(APPCONF=APPCONF) -> ConfigParser:
    """
    Load CONFIG object located at [APPCONF] if exists,
    else create a new one.

    :returns: ConfigParser
    """
    global _CONFIG

    if _CONFIG is None:
        _CONFIG = ConfigParser()
        _CONFIG = ConfigParser(inline_comment_prefixes="#")
        _CONFIG.optionxform = str  # preserve case

    if APPCONF.exists():
        _CONFIG.read_file(APPCONF.open("r"))
        for section in SECTIONS:
            if section not in _CONFIG.sections():
                _CONFIG[section] = {"default": ""}
        return _CONFIG

    _CONFIG["ext"] = SECTIONS["ext"]
    _CONFIG["refdb"] = SECTIONS["refdb"]
    _CONFIG["search-params"] = SECTIONS["search-params"]
    _CONFIG["quant-params"] = SECTIONS["quant-params"]
    return _CONFIG


def write_config(conf: ConfigParser = None, APPCONF=APPCONF) -> None:
    if conf is None:
        conf = get_conf()
    if not APPCONF.parent.exists():
        APPCONF.parent.mkdir()
    with APPCONF.open("w") as f:
        conf.write(f)


@config_app.command("set-ref")
def set_ref(
    name: str,
    file: Path = typer.Argument(".", exists=True, file_okay=True, dir_okay=False),
):
    """
    Set reference fasta database [dir] to attribute [name]
    """
    conf = get_conf()
    conf["refdb"][name] = str(file.resolve())  # does this work for Path objects?
    write_config()


@config_app.command("set-search")
def set_search(
    name: str,
    file: Path = typer.Argument(
        ".",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """
    Set reference fasta database [dir] to attribute [name]
    """
    conf = get_conf()
    conf["search-params"][name] = str(
        file.resolve()
    )  # does this work for Path objects?
    write_config()


@config_app.command("set-quant")
def set_quant(
    name: str,
    file: Path = typer.Argument(
        ".",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """
    Set reference fasta database [dir] to attribute [name]
    """
    conf = get_conf()
    conf["quant-params"][name] = str(file.resolve())  # does this work for Path objects?
    write_config()


@config_app.command("set-dir")
def set_dir(dir: Path = typer.Argument(".", exists=True, file_okay=True)):
    """set_dir directory that contains subdirectories for MASIC, MSFragger, etc

    Update mspcrunner configuration
    """

    conf = get_conf()  # there is an easy way to do this with PyDantic?
    # conf["ext"] = dir
    # TODO look for other executables

    typer.echo("set all. WIP")
    write_config()


@config_app.command("set-msfragger-dir")
def set_msfragger_dir(file: Path = typer.Argument(".", exists=True, file_okay=True)):
    conf = get_conf()
    conf["ext"]["msfragger-executable"] = str(file.resolve())
    # typer.echo("msfragger dir. WIP")
    write_config()


@config_app.command("set-masic-dir")
def set_masic_dir(file: Path = typer.Argument(".", exists=True, file_okay=True)):
    print(file)
    conf = get_conf()
    conf["ext"]["masic-executable"] = str(
        file.resolve()
    )  # does this work for Path objects?
    # typer.echo("masic dir, WIP")
    write_config()


@config_app.command("show")
def show():
    conf = get_conf()
    # import ipdb; ipdb.set_trace()
    for key in conf.sections():
        typer.echo(f"\n{'='*40}\n~~~ {key} ~~~")
        for k, v in conf[key].items():
            typer.echo(f"{k}:\t{v}")
            # typer.echo(k,v)
        typer.echo("-" * 40)


def validate_entry(file: Path, category: str, remove_missing=False):
    """
    validate file
    :type: one of `ext` `refdb` `search-params` `quant-params` (can be expanded)
    """
    if not file.exists():
        logger.warn(f"{file} does not exist")
        return -1
        # if remove_missing:
        #     file.rem

    if category == "ext":
        logger.info(f"{file} {category}")
    if category == "refdb":
        logger.info(f"{file} {category}")
    if category == "search-params":
        logger.info(f"{file} {category}")
    if category == "quant-params":
        logger.info(f"{file} {category}")
    return 0


def two():
    pass


@config_app.command("check")
def check(
    remove: Optional[bool] = typer.Option(False),
):
    """
    check if current config specification is valid
    """
    conf = get_conf()
    for key in conf.sections():
        for name, value in conf[key].items():
            fileobj = Path(value)
            ret_code = validate_entry(fileobj, category=key, remove_missing=remove)
            if ret_code == 0:
                continue
            if ret_code != 0 and remove:
                conf[key].popitem(value)
    write_config()
    # typer.echo(f"{k}:\t{v}")


def get_msfragger_exe():
    conf = get_conf()
    return Path(conf["ext"]["msfragger-executable"]) or None


def get_masic_exe():
    conf = get_conf()
    return Path(conf["ext"]["masic-executable"]) or None
