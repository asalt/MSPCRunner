import logging
from mspcrunner.containers import RunContainer
import pytest
import mspcrunner
from mspcrunner.containers import RunContainer
from mspcrunner.file_finder import FileFinder


PATH = "path"


@pytest.fixture(scope="session")
def tmpdir_session(ext, tmpdir_factory):
    """A tmpdir fixture for the session scope. Persists throughout the pytest session."""
    return tmpdir_factory.mktemp(request.session.name)


def test_filefinder():
    pass


@pytest.fixture(scope="session")
def raw_file(tmp_path_factory):

    f = tmp_path_factory.mktemp("data") / "test.raw"
    f.touch()
    return f


# @pytest.fixture(scope="session")
# def place_file(ext):
#     # temp_path_factory = TempPathFactory()
#     f = temp_path_factory.mktemp(PATH) / f"test.{ext}"
#     f.touch()
#     return f

TEST_CASES = [
    [("raw",), 1],
    [("raw", "pin"), 2],
    [("raw", "pin", "junk"), 2],
    [("raw", "pin", "tsv"), 3],
]

# contents of test_image.py
@pytest.mark.parametrize("exts,num_found", TEST_CASES)
def test_find_some_file(tmp_path_factory, exts, num_found):
    logging.debug(f"{tmp_path_factory}, {exts}, {num_found}")
    path = tmp_path_factory.mktemp("tmp")
    for ext in exts:
        f = path / f"test.{ext}"
        f.touch()

    filefinder = FileFinder()
    results = filefinder.run(path=path)
    assert len(results) == 1
    assert isinstance(results[0], RunContainer)

    runcontainer = results[0]
    assert len(runcontainer._files) == num_found

    # img = load_image(image_file)
    # compute and test histogram


# @pytest.mark.parametrize("exts,num_found", TEST_CASES)
# def test_get_a_file(tmp_path_factory, exts, num_found):
#     filefinder = FileFinder()
#     results = filefinder.run(path=path)