"""
xxRun the pipeline with: python run.py
"""
from glob import glob
from pathlib import Path
from ploomber import DAG
from ploomber.products import File
from ploomber.spec import DAGSpec
from ploomber.tasks import ShellScript

from ploomber.executors import Parallel

# from tasks import get_raw, get_raw_only

import yaml  # this is a good diea
from copy import deepcopy

MASIC_EXE = "<MASIC_EXE>"

PARAMFILE_MASIC = Path("<paramfile_masic>")
OUTPUTDIR_MASIC = Path("<outputdir_masic>")
INPUTFILE_MASIC = Path("<inputfile_masic>")


def _merge_nested_dicts(d1: dict, d2: dict) -> dict:
    d1 = deepcopy(d1)  # we do not want to mutate the original dict
    # https://stackoverflow.com/a/22093909
    result = {
        key: value.update(d2[key]) or value if key in d2 else value
        for key, value in d1.items()
    }
    return result


def calculate_masic_output(p: Path) -> Path:
    # return Path("raw") / f"{p.stem}_SICstats.txt"
    # return p.parent / f"{p.stem}_SICstats.txt"  # customize here
    return Path(f"{p.stem}_SICstats.txt")  # customize here


_env_obj = None


LOCAL_ENV = Path("env.yaml")


def get_env(local_env=LOCAL_ENV):
    global _env_obj
    if _env_obj is None:
        with open(local_env) as f:
            env = yaml.safe_load(f)
        _env_obj = env
    return _env_obj


def set_env(orig_env: dict, raw_file: Path) -> dict:
    masic_output = calculate_masic_output(raw_file)

    dynamic_params = {
        "masic": {
            "inputfile": raw_file,  # file name
            "output": masic_output,  # file name
        },
    }
    final_env = _merge_nested_dicts(orig_env, dynamic_params)
    # test
    final_env["masic"]["paramfile"] = Path(final_env["masic"]["paramfile"])
    return final_env


def run(raw_files: list):
    print("hello world")
    # grab the files from the folder and run the pipeline

    # _d = get_glob_directory()
    # raw_files = get_raw_only(_d)
    # raw_files = get_raw(_d)

    orig_env = get_env(LOCAL_ENV)

    # we build a new DAG to accumulate all masic tasks
    dag = DAG(executor=Parallel(processes=8, print_progress=True))
    import ipdb

    ipdb.set_trace()

    # ============================================================
    def add_task(raw_file):
        final_env = set_env(orig_env, raw_file)

        print(f"adding env: {final_env}")

        expected_output = calculate_masic_output(raw_file)
        _masic_output_dir = orig_env["masic"]["outputdir"]
        product = File(Path(_masic_output_dir) / Path(expected_output))

        ShellScript(
            Path(
                "./scripts/run_masic.sh"
            ),  # this is relative to the current file within /mspcrunner/src/mspcrunner/ploomber/
            dag=dag,
            product=product,
            name=raw_file,
            params=final_env,
        )

    # ============================================================

    # walk
    [x for x in map(add_task, raw_files)]
    dag.build(force=True)
