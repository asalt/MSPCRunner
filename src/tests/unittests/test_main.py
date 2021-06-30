import os
from pathlib import Path

import pytest

from mspcrunner.main import set_outdir


def test_set_outdir():

    path = Path(__file__)
    outdir = Path.home()

    assert set_outdir(outdir=None, path=None).resolve() == Path(os.getcwd()).resolve()
    assert set_outdir(outdir=None, path=path).resolve() == path.resolve()
    assert set_outdir(outdir=outdir, path=path).resolve() == outdir.resolve()
