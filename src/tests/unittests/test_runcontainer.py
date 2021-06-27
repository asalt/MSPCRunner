from pathlib import Path

import pytest

import mspcrunner
from mspcrunner import containers
from mspcrunner.containers import RunContainer


def test_stem_default_none():
    run_container = RunContainer(stem=None)
    assert run_container.stem is None


def test_stem_add_irrelevant_file():
    run_container = RunContainer(stem=None)
    run_container.add_file(Path("not-interesting-file"))
    assert run_container.stem is None


def test_add_spectrafile():
    run_container = RunContainer()
    testfile = Path("testfile.raw")
    run_container.add_file(testfile)
    assert run_container.get_file("spectra") == testfile
    assert run_container.stem is not None




def test_add_raw_then_mzml():
    run_container = RunContainer()

    rawfile = Path("testfile.raw")
    mzmlfile = Path("testfile.mzML")

    run_container.add_file(rawfile)
    assert run_container.get_file("spectra") == rawfile
    assert run_container.get_file("raw") == rawfile

    run_container.add_file(mzmlfile)
    assert run_container.get_file("spectra") == mzmlfile
    assert run_container.get_file("raw") == rawfile
    assert run_container.stem is not None


def test_stem_add_irrelevant_filexxx():
    run_container = RunContainer(stem=None)
    testfiles = [
        "12345_1_1_aofejf",
        "12345_1_1_aofejf",
        "12345_1_1_aofejf",
        "12345_1_1_aofejf",
    ]
    # run_container[]
    assert run_container.stem is None

def test_rootdir_set(tmp_path):
    f1 = tmp_path / 'testfile.raw'
    f2 = tmp_path / 'testfile.tsv'

    run_container = RunContainer()
    run_container.add_file(f1)
    run_container.add_file(f2)
    assert run_container.rootdir == tmp_path

def test_rootdir_valueerror(tmp_path):
    f1 = tmp_path / 'testfile.raw'
    f2 = tmp_path / 'loc' /'testfile.tsv'

    run_container = RunContainer()
    run_container.add_file(f1)
    run_container.add_file(f2)
    with pytest.raises(ValueError):
         run_container.rootdir