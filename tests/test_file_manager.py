from quack.utils import get_quack_home
from pathlib import Path


def test_home_directory():
    assert get_quack_home() == Path.home() / '.quack_data'
