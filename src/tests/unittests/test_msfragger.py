import logging

import mspcrunner
from mspcrunner import logger
from mspcrunner.MSFragger import get_exe, MSFragger
from mspcrunner.predefined_params import Predefined_Search, PREDEFINED_SEARCH_PARAMS

__PARAMFILE__ = PREDEFINED_SEARCH_PARAMS[list(PREDEFINED_SEARCH_PARAMS.keys())[0]]


def test_set_param():

    # test when no param file sepcified
    msfragger = MSFragger(receiver=None)
    assert msfragger.set_param("calibrate_mass", 2) == -1

    msfragger = MSFragger(receiver=None, paramfile=__PARAMFILE__)
    assert msfragger.set_param("calibrate_mass", 2) == 0

    # TODO return as int?
    assert str(msfragger._config["top"]["calibrate_mass"]) == str(2)


def test_no_exist_until_write(tmp_path_factory):
    msfragger = MSFragger(receiver=None, paramfile=__PARAMFILE__)
    assert msfragger.paramfile is None
    msfragger.write_config()
    assert msfragger.paramfile is not None
    logging.debug(f"{msfragger.paramfile.resolve()}")

    msfragger.paramfile.unlink(missing_ok=True)
    # assert msfragger.set_param("calibrate_mass", 2) == -1
