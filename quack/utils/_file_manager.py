import os
import urllib
import urllib.request
from pathlib import Path
from tqdm import tqdm


class TqdmPBar(tqdm):


def download_file(url: str, filename: str, exist_ok: bool = False) -> None:
    """Download a file from an specific URL and rename the local base filename.

    Args:
        url (str): URL that the file will be downloaded from. 
        filename (str): filename of the local archive that will be downloaded.
        exist_ok (bool, optional): Flag to download the file even if it is already
            exists on local filesystem. Defaults to False.
    """
    def _progress_bar(block_num, bs, size):
        total_size_mb = '%.2f MB' % (size / 1e6)
        current_size_mb = '%.2f MB' % ((block_num * bs) / 1e6)
        print('\rdownloaded %s / %s' % (current_size_mb, total_size_mb), end='')

    if os.path.exists(filename) and exist_ok is False:
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print('Downloading from %s' % url)
    urllib.request.urlretrieve(url, filename=filename, reporthook=_progress_bar)
    print('')


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
