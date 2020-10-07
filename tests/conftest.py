import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_directory() -> Path:
    return Path(os.path.dirname(__file__)) / "fixures"
