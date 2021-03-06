import os
from enum import Enum
from pathlib import Path
import re
import pkg_resources

from .config import get_conf

PKG_PROVIDER = pkg_resources.get_provider("mspcrunner")
BASEDIR = Path(PKG_PROVIDER.module_path).parent.parent

# BASEDIR = Path(os.path.split(__file__)[0]).resolve()
# BASEDIR = pkg_resources.resource_filename


def get_dir(s: str) -> Path:
    out = (Path(os.path.split(__file__)[0]).parent.parent / s).resolve()
    return out


class Param(str, Enum):
    pass


def param_obj_creator(name: str, d: dict):
    return Param(name, {str(v): k for k, v in d.items()})


# return cls(cls.__name__, d)

PARAMDIR = (Path(os.path.split(__file__)[0]).parent.parent / "params").resolve()

PARAMDIR = get_dir("params")
TEMPLATEDIR = get_dir("templates")

conf = get_conf()


PREDEFINED_REFSEQ_PARAMS = {
    "hs2020": "/mnt/e/reference_databases/2020-11-23-decoys-contam-gpGrouper_Homo_sapiens_2020_03_24_refseq_GCF_000001405.39_GRCh38.p13_protein.fa.fas",
    "hsmm2020": "/mnt/e/reference_databases/2020-10-29-decoys-contam-gpGrouper_HS_MM_2020_03_24_refseq_GCF_000001405.39_GRCh38.p13_GCF_000001635.26_GRCm38.p6.fa.fas",
}

# for k,v in conf['refdb'].items():
if "refdb" in conf.sections():
    PREDEFINED_REFSEQ_PARAMS.update({k: v for k, v in conf["refdb"].items() if v})
# PREDEFINED_REFSEQ_PARAMS.update(conf['refdb'].items())


Predefined_RefSeq = param_obj_creator(
    "Predefined_RefSeq",
    PREDEFINED_REFSEQ_PARAMS
    # {v: k for k, v in PREDEFINED_REFSEQ_PARAMS.items()}
)


# OTIT = "OTIT"
# OTIT_hs = "OTIT-hs"
# OTOT = "OTOT"
# TMT6_OTOT = "TMT6-OTOT"
# TMT6_OTOT_PHOS = "TMT6-OTOT-PHOS"
# TMT6_OTOT_QC = "TMT6-OTOT-QC"
# TMT16_OTOT = "TMT16-OTOT"
# TMT16_OTOT_QC = "TMT16-OTOT-QC"


# PREDEFINED_SEARCH_PARAMS = {
#     #'OTIT' : Path('../params/MSfragger_OTIT.conf'),
#     "OTIT-hs": PARAMDIR / Path("MSFragger_OTIT_hs.conf"),
#     "OTOT": PARAMDIR / Path("MSFragger_OTOT.conf"),
#     "TMT6-OTOT": PARAMDIR / Path("MSFragger_TMT6_OTOT.conf"),
#     "TMT6-OTOT-PHOS": PARAMDIR / Path("MSFragger_TMT6_OTOT_PHOS.conf"),
#     "TMT6-OTOT-QC": PARAMDIR / Path("MSFragger_TMT6_OTOT_QC.conf"),
#     "TMT11-OTOT-QC": PARAMDIR / Path("MSFragger_TMT6_OTOT_QC.conf"),
#     "TMT16.OTOT": PARAMDIR / Path("MSFragger_OTOT.conf"),
#     "TMT16-OTOT-QC": PARAMDIR / Path("MSFragger_OTOT.conf"),
# }
PREDEFINED_SEARCH_PARAMS = {}

if "search-params" in conf.sections():
    PREDEFINED_SEARCH_PARAMS.update(
        {k: Path(v) for k, v in conf["search-params"].items() if v}
    )

Predefined_Search = param_obj_creator("Predefined_Search", PREDEFINED_SEARCH_PARAMS)


# LF = "LF"
# TMT10 = "TMT10"
# TMT11 = "TMT11"
# TMT16 = "TMT16"


PREDEFINED_QUANT_PARAMS = {
    #'OTIT' : Path('../params/MSfragger_OTIT.conf'),
    # "TMT11": PARAMDIR / Path("MASIC_TMT11.xml"),
    "LF": PARAMDIR / "MASIC_LF_10ppm.xml",
    "TMT10": PARAMDIR / "MASIC_TMT10_10ppm_ReporterTol0.003Da.xml",
    "TMT11": PARAMDIR / "MASIC_TMT11_10ppm_ReporterTol0.003Da.xml",
    "TMT16": PARAMDIR / Path("MASIC_TMT16.xml"),
}

if "quant-params" in conf.sections():
    PREDEFINED_QUANT_PARAMS.update(
        {k: Path(v) for k, v in conf["quant-params"].items() if v}
    )
Predefined_Quant = param_obj_creator("Predefined_Quant", PREDEFINED_QUANT_PARAMS)


class Predefined_gpG(str, Enum):
    default = PARAMDIR / "gpgrouper.conf"


class RMD_OUT_FORMAT(str, Enum):
    html = "html"
    pdf = "pdf"


PREDEFINED_RMD_TEMPLATES = {
    "TMTreport": TEMPLATEDIR / Path("TMTreport_a1.Rmd"),
}

RMD_TEMPLATES = param_obj_creator("RMD_TEMPLATES", PREDEFINED_RMD_TEMPLATES)
