"""PSM_Concat input is RunContainers

It concatenates all

"""

from pathlib import Path
import logging

import pytest
from mspcrunner.psm_concat import PSM_Concat
from mspcrunner.containers import RunContainer


@pytest.fixture(scope="module")
def something(tmp_path_factory):
    ...
