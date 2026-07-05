from quack import get_version

def test_get_version():
    assert get_version() == "0.1.0"
