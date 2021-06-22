"""
"""

import sys
from pathlib import Path
import subprocess

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(f"Usage {__file__} path")
        sys.exit(0)

    path = Path(sys.argv[1])
    for f in path.glob("*pin"):

        print(f, f.stem)
        subprocess.run(
            [
                "mokapot",
                "--seed",
                "8888",
                "--file_root",
                f"{f.stem}",
                f.name,
                "--max_iter",
                "5",
            ]
        )
