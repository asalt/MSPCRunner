# TODO use fixtures?
from pathlib import Path

import pytest

import mspcrunner
from mspcrunner import containers
from mspcrunner.containers import AbstractContainer, RunContainer, SampleRunContainer


@pytest.fixture
def two_valid_files():
    rawfile = Path("testfile.raw")
    mzmlfile = Path("testfile.mzML")


def test_stem_default_none():
    run_container = RunContainer(stem=None)
    assert run_container.stem is None


def test_stem_nofiles():
    run_container = RunContainer(stem=None)
    assert run_container.stem == None
    assert run_container.rootdir == None
    assert run_container.update_files() is None


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


def test_property_reset():

    f1 = Path("12345_1_1_1file.raw")
    f2 = Path("12345_1_1_2file.mzML")

    run_container = RunContainer(stem=None)
    run_container.add_file(f1)
    assert run_container.stem == f1.stem

    run_container.add_file(f2)
    assert run_container._stem is None


def test_stem_creation():
    run_container = RunContainer(stem=None)

    f1 = Path("12345_1_1_1file.raw")
    f2 = Path("12345_1_1_2file.mzML")

    run_container.add_file(f1)
    run_container.add_file(f2)

    # print(run_container.stem)
    assert run_container.stem == "12345_1_1"

    run_container.add_file(Path("irrelevant-file"))
    assert run_container.stem == "12345_1_1"


def test_rootdir_set(
    tmp_path,
):
    f1 = tmp_path / "testfile.raw"
    f2 = tmp_path / "testfile.tsv"

    run_container = RunContainer()
    run_container.add_file(f1)
    run_container.add_file(f2)
    assert run_container.rootdir == tmp_path


def test_relocate(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    f1 = d / "testfile.raw"
    f1.touch()
    assert f1.exists()
    assert d.exists()

    run_container = RunContainer()
    run_container.add_file(f1)

    dnew = tmp_path / "new"
    dnew.mkdir()
    assert dnew.exists()
    # print(dnew, list(dnew.glob("*")))
    run_container.relocate(dnew)


# def test_rootdir_valueerror(tmp_path):
#    ""
#    f1 = tmp_path / "testfile.raw"
#    f2 = tmp_path / "loc" / "testfile.tsv"
#
#    run_container = RunContainer()
#    run_container.add_file(f1)
#    run_container.add_file(f2)
#    assert run_container.rootdir is None
#    # with pytest.raises(ValueError):
#    #     run_container.rootdir


def test_updatefile_noroot():
    ...
    # ...run_container = RunContainer()
    # ...run_container._rootdir = None
    # assert run_container.update_files()


def test_equal_with_no_files():
    assert RunContainer(rootdir=Path(".")) == RunContainer(rootdir=Path(".aefaef"))


def test_not_equal_with_different_files():
    rc1 = RunContainer()
    rc2 = RunContainer()
    rc1.add_file("test.raw")
    assert rc1 != rc2

    rc2.add_file("test.raw")
    assert rc1 == rc2

    rc2.add_file("test.mzML")
    assert rc1 != rc2


def test_not_equal_different_containers():
    assert RunContainer() != SampleRunContainer()


def test_not_equal_different_containers():
    assert RunContainer() != SampleRunContainer()
