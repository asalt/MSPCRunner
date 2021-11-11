"""
"""
import dagster
import dagster as d
from dagster import job, op
from dagster.core.definitions.preset import PresetDefinition
import typer
import typing as ty
from typing import Optional
from pathlib import Path

from mspcrunner.predefined_params import (
    Predefined_RefSeq,
    Predefined_Search,
    Predefined_Quant,
)


# @d.solid()
@op
def get_config():
    return list(Predefined_RefSeq.values())[0]


# @dagster.solid()
@op
def search(
    context,
    # preset=dagster.InputDefinition(None, dagster_type=dagster.Optional),
    # preset: Predefined_Search = typer.Option(None, case_sensitive=False),
    # paramfile: Optional[Path] = typer.Option(None),
    # refseq: Predefined_RefSeq = typer.Option(None),
    # local_refseq: Optional[Path] = typer.Option(None),
    # calibrate_mass: Optional[int] = typer.Option(default=1, min=0, max=2),
    # ramalloc: Optional[int] = typer.Option(
    #     default=10, help="Amount of memory (in GB) for msfragger"
    # ),
    # msfragger_conf: Optional[Path] = typer.Option(MSFRAGGER_DEFAULT_CONF),
) -> int:
    # paramfile = context.solid_config["paramfile"]
    # context.log.info(f"paramfile: {paramfile}")
    context.op_config.get("something important")

    return 1


# @dagster.pipeline(config_schema=dagster.Field(dagster.Any))
# @dagster.pipeline(
#     preset_defs=[
#         d.PresetDefinition(
#             name="test",
#             run_config={"solids": {"search": {"config": {"paramfile": "file2"}}}},
#             # "solid_selection": [search],
#         )
#     ]
# )
@op
def display_results(context, func_name=None, res=None):
    context.log.info("f {func_name} ended with code {res}")


@job
def search_pipeline():
    # conf = get_config()
    search_res = search()
    func_name = search.__name__
    display_results(res=search_res)


@dagster.repository
def search_repository():
    # return [
    #     search_pipeline(),
    #     # search(),
    # ]
    return {
        "pipelines": {
            "search_pipeline": lambda: search_pipeline,
        }
    }


# this is how we can run a job "manually"
if __name__ == "__main__":
    result = search.execute_in_process()
