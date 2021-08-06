import logging
from mspcrunner import worker
from mspcrunner.containers import RunContainer, SampleRunContainer
from mspcrunner.worker import Worker
import pytest


@pytest.fixture()
def example_runcontainer(tmp_path_factory):
    rc = RunContainer()
    f = tmp_path_factory.mktemp("tmp") / "12345_1_1_test.raw"
    f.touch()
    rc.add_file(f)
    return rc


# MATCHING_NAME_CLS = [
#     ("add_runcontainer", RunContainer()),
#     ("add_sampleruncontainer", SampleRunContainer()),
# ]


def test_worker_add_run_container():
    worker = Worker()
    assert len(worker._runcontainers) == 0
    worker.add_runcontainer(RunContainer())
    assert len(worker._runcontainers) == 1


TEST_CASES = [
    ("add_runcontainer", SampleRunContainer()),
    ("add_sampleruncontainer", RunContainer()),
]


@pytest.mark.parametrize("method,cls", TEST_CASES)
def test_worker_add_wrong_container(method, cls):
    worker = Worker()
    with pytest.raises(ValueError):
        getattr(worker, method)(cls)
        # worker.add_runcontainer(SampleRunContainer())


def xxtest_worker1():
    worker = Worker()

    rc1 = RunContainer().add_file("test.raw")
    from copy import deepcopy

    rc2 = deepcopy(rc1)
    rc2.add_file("test.mzML")

    worker.add_runcontainer()

    rc1.add_file("test.raw")
    assert rc1 != rc2

    rc2.add_file("test.raw")
    assert rc1 == rc2
    pass