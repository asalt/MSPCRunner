import os
from enum import Enum
from pathlib import Path


BASEDIR = Path(os.path.split(__file__)[0]).resolve()
PARAMDIR = (Path(os.path.split(__file__)[0]).parent.parent / "params").resolve()


class Predefined_Search(str, Enum):

    OTIT = ("OTIT",)
    OTIT_hs = "OTIT-hs"
    OTOT = "OTOT"
    TMT6_OTOT = "TMT6-OTOT"
    TMT6_OTOT_QC = "TMT6-OTOT-QC"
    TMT16_OTOT = "TMT16-OTOT"
    TMT16_OTOT_QC = "TMT16-OTOT-QC"


PREDEFINED_SEARCH_PARAMS = {
    #'OTIT' : Path('../params/MSfragger_OTIT.conf'),
    "OTIT-hs": PARAMDIR / Path("MSFragger_OTIT_hs.conf"),
    "OTOT": PARAMDIR / Path("MSFragger_OTOT.conf"),
    "TMT6-OTOT": PARAMDIR / Path("MSFragger_TMT6_OTOT.conf"),
    "TMT6-OTOT-QC": PARAMDIR / Path("MSFragger_TMT6_OTOT_QC.conf"),
    "TMT11-OTOT-QC": PARAMDIR / Path("MSFragger_TMT6_OTOT_QC.conf"),
    "TMT16.OTOT": PARAMDIR / Path("MSFragger_OTOT.conf"),
    "TMT16-OTOT-QC": PARAMDIR / Path("MSFragger_OTOT.conf"),
}


class Predefined_Quant(str, Enum):
    TMT11 = "TMT11"
    TMT16 = "TMT16"


PREDEFINED_QUANT_PARAMS = {
    #'OTIT' : Path('../params/MSfragger_OTIT.conf'),
    # "TMT11": PARAMDIR / Path("MASIC_TMT11.xml"),
    "TMT11": PARAMDIR / "MASIC_TMT11_10ppm_ReporterTol0.003Da.xml",
    "TMT16": PARAMDIR / Path("MASIC_TMT16.xml"),
}
