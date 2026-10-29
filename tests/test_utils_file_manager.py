# tests/test_utils_file_manager.py
import hashlib
import urllib.request

import pytest

from quack.utils import check_file_hash, download_file, file_hash


@pytest.fixture
def server(monkeypatch):
  """Replaces `urlretrieve` with a function that writes `server.content`."""

  class Server:
    content = b"hello quack"
    calls = 0

  def fake_urlretrieve(url, filename, reporthook=None):
    Server.calls += 1
    with open(filename, "wb") as fh:
      fh.write(Server.content)

  monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
  return Server


def sha256(data: bytes) -> str:
  return "sha256:" + hashlib.sha256(data).hexdigest()


class TestFileHash:
  def test_file_hash(self, tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert file_hash(str(path)) == hashlib.sha256(b"abc").hexdigest()
    assert file_hash(str(path), "md5") == hashlib.md5(b"abc").hexdigest()

  def test_check_file_hash(self, tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert check_file_hash(str(path), sha256(b"abc"))
    assert check_file_hash(str(path), sha256(b"abc").upper().replace("SHA256", "sha256"))
    assert not check_file_hash(str(path), sha256(b"abd"))
    assert not check_file_hash(str(tmp_path / "missing"), sha256(b"abc"))

  def test_check_file_hash_requires_algorithm_prefix(self, tmp_path):
    with pytest.raises(ValueError, match="algorithm"):
      check_file_hash(str(tmp_path / "f.bin"), hashlib.sha256(b"abc").hexdigest())


class TestDownloadFile:
  def test_creates_folders_and_uses_cache(self, tmp_path, server):
    target = tmp_path / "a" / "b" / "file.bin"
    download_file("https://example.org/file.bin", str(target))
    download_file("https://example.org/file.bin", str(target))
    assert target.read_bytes() == b"hello quack"
    assert server.calls == 1

  def test_exist_ok_downloads_again(self, tmp_path, server):
    target = tmp_path / "file.bin"
    download_file("https://example.org/file.bin", str(target))
    download_file("https://example.org/file.bin", str(target), exist_ok=True)
    assert server.calls == 2

  def test_verified_download(self, tmp_path, server):
    target = tmp_path / "file.bin"
    download_file("https://example.org/file.bin", str(target), checksum=sha256(b"hello quack"))
    assert target.exists()

  def test_corrupted_cache_is_downloaded_again(self, tmp_path, server):
    target = tmp_path / "file.bin"
    target.write_bytes(b"truncated")
    download_file("https://example.org/file.bin", str(target), checksum=sha256(b"hello quack"))
    assert target.read_bytes() == b"hello quack"
    assert server.calls == 1

  def test_mismatching_download_is_removed(self, tmp_path, server):
    target = tmp_path / "file.bin"
    with pytest.raises(ValueError, match="does not match its checksum"):
      download_file("https://example.org/file.bin", str(target), checksum=sha256(b"other content"))
    assert not target.exists()
