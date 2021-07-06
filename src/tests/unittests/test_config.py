from configparser import ConfigParser
from pathlib import Path

from configparser import ConfigParser

import pytest

import mspcrunner
from mspcrunner import containers
from mspcrunner.containers import RunContainer

# from
from mspcrunner import config

# from mspcrunner.config import get_conf

# MOCK appdir
# config.APPDIR = ""


@pytest.fixture
def CONFIG():
    c = ConfigParser()
    for key, vals in config.SECTIONS.items():
        c[key] = vals
    return c


def test_load_new_config(tmp_path):
    NOFILE = tmp_path / "does-not-exist"
    result = config.load_config(NOFILE)
    assert isinstance(result, ConfigParser)


def test_set_appconf(tmp_path):
    appconf = tmp_path / "test.conf"
    # _ = config.load_config(appconf)
    config.write_config(APPCONF=appconf)
    assert appconf.exists()


def test_write_config(tmp_path):
    # d.mkdir()
    conf_path = tmp_path / "test.conf"
    conf = config.load_config(conf_path)
    # print(conf_path, conf_path.exists(), d.exists() )
    config.write_config(conf, conf_path)


def test_existing_config(tmp_path, CONFIG):
    tmp_path.mkdir(exist_ok=True, parents=True)
    existing_conf = tmp_path / "mspcrunner.conf"
    existing_conf.touch()
    # existing_conf.write
    # CONFIG['ext']['test'] = 'test'

    # print(CONFIG)

    CONFIG["ext"]["test"] = "test"
    with existing_conf.open("w") as f:
        CONFIG.write(f)

    c = config.load_config(existing_conf)
    assert c["ext"]["test"] == "test"
