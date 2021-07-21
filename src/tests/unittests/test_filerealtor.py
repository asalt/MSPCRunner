from pathlib import Path
import logging

import pytest
import mspcrunner
from mspcrunner import file_finder

from mspcrunner import commands
from mspcrunner.commands import FileRealtor
from mspcrunner.containers import RunContainer


@pytest.fixture(scope="session")
def image_file(tmp_path_factory):
    img = compute_expensive_image()
    fn = tmp_path_factory.mktemp("data") / "img.png"
    img.save(fn)
    return fn


@pytest.fixture(scope="module")
def example_runcontainer(tmp_path_factory):
    rc = RunContainer()
    f = tmp_path_factory.mktemp("tmp") / "12345_1_1_test.raw"
    f.touch()
    rc.add_file(f)
    return rc


def test_example_runcontainer(example_runcontainer):
    # logging.debug(example_runcontainer._files)
    assert len(example_runcontainer._files) == 1


def test_stem():
    pass


def test_one(example_runcontainer):
    filerealtor = FileRealtor()

    # basepath = Path(example_runcontainer.stem)
    basepath = Path(example_runcontainer._files[0]).parent

    # logging.error(basepath)
    # logging.debug("=" * 6)
    # logging.error([x for x in basepath.glob("*")])
    # logging.debug("=" * 6)

    assert "12345_1_1_test.raw" in [x.name for x in basepath.glob("*")]

    filerealtor.run((example_runcontainer,))

    assert len([x for x in basepath.glob("*") if x.is_file()]) == 0

    # logging.error(basepath)
    # logging.error(
    #     example_runcontainer.stem,
    # )
    # logging.error(
    #     basepath / example_runcontainer.stem,
    # )
    # logging.error([x for x in (basepath / "12345_1_1").glob("*")])

    assert len([x for x in (basepath / "12345_1_1").glob("*")]) == 1
