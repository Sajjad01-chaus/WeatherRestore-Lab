from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.prepare_mirror_data import numbered_files, verified_extract
from src.inputs import ValidationError
from src.records import sha256


def test_archive_validation_rejects_escape_and_wrong_hash(tmp_path: Path) -> None:
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('../escape.txt', 'data')
    with pytest.raises(ValidationError, match='checksum'):
        verified_extract(archive, tmp_path / 'wrong', '0' * 64)
    with pytest.raises(ValidationError, match='Unsafe archive path'):
        verified_extract(archive, tmp_path / 'unsafe', sha256(archive))
    assert not (tmp_path / 'unsafe').exists()
    assert not (tmp_path / 'escape.txt').exists()


def test_verified_archive_never_overwrites(tmp_path: Path) -> None:
    archive = tmp_path / 'good.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('nested/image.png', b'test fixture')
    destination = tmp_path / 'extracted'
    verified_extract(archive, destination, sha256(archive))
    assert (destination / 'nested/image.png').read_bytes() == b'test fixture'
    with pytest.raises(ValidationError, match='overwrite'):
        verified_extract(archive, destination, sha256(archive))


def test_numeric_pairing_rejects_ambiguous_ids(tmp_path: Path) -> None:
    (tmp_path / 'rain-001.png').write_bytes(b'a')
    (tmp_path / 'rain-1.png').write_bytes(b'b')
    with pytest.raises(ValidationError, match='duplicate numeric'):
        numbered_files(tmp_path, 'rain')
