#! /usr/bin/env python

import sys
from pathlib import Path
import pandas as pd

from mokapot import read_pin

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f'Usage {__file__} path')
        sys.exit(0)

    path = Path(sys.argv[1])
    for f in path.glob("*pin"):

        df = read_pin(str(f), to_df=True)
        df['Phos'] = 0
        df.loc[ df['Peptide'].str.contains('[79.966', regex=False), 'Phos' ] = 1
        df = df[[x for x in df if x != 'Proteins'] + ['Proteins']]
        df.to_csv(f, sep='\t', index=False)
        print(f"Added phos bool to {f}")


