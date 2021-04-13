import os 
from enum import Enum
from pathlib import Path


BASEDIR = Path(os.path.split(__file__)[0])


class Predefined_Search(str, Enum):

    OTIT = 'OTIT',
    OTIT_hs = 'OTIT-hs'
    OTOT = 'OTOT'
    TMT6_OTOT = 'TMT6-OTOT'
    TMT6_OTOT_QC = 'TMT6-OTOT-QC'
    TMT16_OTOT = 'TMT16-OTOT'
    TMT16_OTOT_QC = 'TMT16-OTOT-QC'

PREDEFINED_SEARCH_PARAMS = {
    #'OTIT' : Path('../params/MSfragger_OTIT.conf'),
    'OTIT-hs' : BASEDIR / Path('../../params/MSfragger_OTIT_hs.conf'),
    'OTOT' : BASEDIR / Path('../../params/MSfragger_OTOT.conf'),
    'TMT6-OTOT' : BASEDIR / Path('../../params/MSfragger_OTOT.conf'),
    'TMT6-OTOT-QC' : BASEDIR / Path('../../params/MSfragger_TMT6_OTOT_QC.conf'),
    'TMT11-OTOT-QC' : BASEDIR / Path('../../params/MSfragger_TMT6_OTOT_QC.conf'),
    'TMT16.OTOT' : BASEDIR / Path('../../params/MSfragger_OTOT.conf'),
    'TMT16-OTOT-QC' : BASEDIR / Path('../../params/MSfragger_OTOT.conf'),
}

class Predefined_Quant(str, Enum):
    TMT11 = 'TMT11'
    TMT16 = 'TMT16'

PREDEFINED_QUANT_PARAMS = {
    #'OTIT' : Path('../params/MSfragger_OTIT.conf'),
    'TMT11' : BASEDIR / Path('../params/MASIC_TMT11.xml'),    
    'TMT16' : BASEDIR / Path('../params/MASIC_TMT16.xml'),
}
