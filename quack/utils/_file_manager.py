import os
from pathlib import Path


def get_quack_home() -> Path:
    """
    Get the home directory of QuacK, this is the folder where QuacK saves the
    datasets and metadata, such as the downloaded datasets from online repositories.

    The directory is `~/.quack_data/`

    Returns:
        home (str): base directory of quack data is this machine.
    """
    home = Path.home() / '.quack_data'
    os.makedirs(home, exist_ok=True)
    return home
