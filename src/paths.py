"""
Exposes common paths useful for manipulating datasets and generating figures.

"""
from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[1].absolute()

# Absolute path to the `src` folder
src = root / "src"

# Absolute path to the `src/data` folder (contains datasets)
data = root / "data"

# Absolute path to the output folder
output = root / "output"
