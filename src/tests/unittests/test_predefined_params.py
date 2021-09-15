from pathlib import Path
from enum import Enum, EnumMeta
import logging

import pytest

import mspcrunner
from mspcrunner import predefined_params
from mspcrunner.containers import RunContainer
import pkg_resources

PKG_PROVIDER = pkg_resources.get_provider("mspcrunner")
BASEDIR = Path(PKG_PROVIDER.module_path).parent.parent

from mspcrunner.predefined_params import (
    PREDEFINED_QUANT_PARAMS,
    PREDEFINED_REFSEQ_PARAMS,
    PREDEFINED_SEARCH_PARAMS,
    Predefined_gpG,
    Predefined_Quant,
    Predefined_RefSeq,
    Predefined_Search,
)

# from
from mspcrunner import config


def test_get_dir():
    """
    should return full directory in "main" folder
    """
    res = predefined_params.get_dir("test")
    # print(pkg_resources)
    expected_result = BASEDIR / "test"
    logging.debug(f"{PKG_PROVIDER}")
    logging.debug(f"{BASEDIR}")
    logging.debug(f"{res} -- {expected_result}")
    assert expected_result == res


def test_creator():
    ret = predefined_params.param_obj_creator("hello", dict())
    assert isinstance(ret, EnumMeta)
