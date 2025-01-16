# src/__init__.py
# This top-level __init__ can import frequently used items to make them accessible directly from `src`.
# For example, import the config variables at the top level.

from .config import (
    RAW_BED_FILE,
    RAW_FASTA_FILE,
    PREPARED_DATA_CSV,
    FEATURE_DATA_CSV,
    RANDOM_SEED
)

# This way, I can do: from src import PREPARED_DATA_CSV
