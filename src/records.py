from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30))


def timestamp() -> str:
    return datetime.now(IST).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(encoded, encoding='utf-8')
    os.replace(temporary, path)


def event(path: Path, name: str, **fields: Any) -> None:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'at': timestamp(), 'event': name, **fields}, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def revision(directory: Path) -> str | None:
    try:
        result = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ['numpy', 'Pillow', 'scipy', 'scikit-image', 'pandas', 'streamlit', 'torch', 'einops']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'source_revision': revision(Path.cwd()),
            'power_mode': 'unknown', 'other_gpu_workloads': 'unknown'}
