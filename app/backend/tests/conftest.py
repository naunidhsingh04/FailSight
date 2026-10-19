import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.backend.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(create_app()) as c:
        yield c
