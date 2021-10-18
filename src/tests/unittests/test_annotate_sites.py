from pathlib import Path
import logging
from mspcrunner import logger
import pytest
import mspcrunner
from mspcrunner import inhouse_modisite
from mspcrunner.containers import SampleRunContainer


inhouse_modisite.AnnotateSite


@pytest.fixture(scope="module")
def example_sampleruncontainer(tmp_path_factory):
    rc = SampleRunContainer()
    f = tmp_path_factory.mktemp("tmp") / "12345_1_6_test_e2g_QUAL_.tsv"
    f.touch()
    rc.add_file(f)
    return rc


def test_create_from_factory(example_sampleruncontainer):
    factory = inhouse_modisite.AnnotateSite(
        receiver=None,
        name="TEST INHOUSE MODISITE ANNOTATOR",
        containers=[example_sampleruncontainer],
    )
    task = factory.create()
    task = next(task)
    # logging.debug(task)
    assert len(task.containers) == 1


def test_create_from_factory_obj(example_sampleruncontainer):
    factory = inhouse_modisite.AnnotateSite(
        receiver=None,
        name="TEST INHOUSE MODISITE ANNOTATOR",
        # containers=[example_sampleruncontainer],
    )
    task = factory.create(sampleruncontainers=[example_sampleruncontainer])
    task = next(task)
    logging.debug(example_sampleruncontainer)
    # logging.debug(task)
    assert len(task.containers) == 1


def test_cmdline_gen(example_sampleruncontainer):
    #  example expected output ['python', PosixPath('somepath/MSPCRunner/src/mspcrunner/ext/annotate_site/annotate_protein.py'), '--all-genes', '--noplot', '--cores', 1, '--fasta', None, '--psms', '12345_1_6']
    task = inhouse_modisite.AnnotateSite(receiver=None).create(
        sampleruncontainers=[example_sampleruncontainer]
    )
    task = next(task)
    task.CMD
    assert isinstance(task.CMD[1], Path)
    assert "annotate_protein.py" in task.CMD[1].name
    assert "--all-genes" in task.CMD
    assert "--noplot" in task.CMD
