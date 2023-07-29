import os

import pytest
from jwt import PyJWKClient
from starlette.datastructures import Headers
from starlette.requests import Request

from tests.auth_mock import mock_jwk


os.environ["PORTAL_DB_USERNAME"] = "test-username"
os.environ["PORTAL_DB_PASSWORD"] = "test-password"


@pytest.fixture(autouse=True)
def mock_auth0_jwks(monkeypatch):
    def mockreturn(self):
        return mock_jwk

    monkeypatch.setattr(
        PyJWKClient,
        "fetch_data",
        mockreturn,
    )


@pytest.fixture
def mock_signing_key():
    def mock_pyjwk(**kwargs):
        return type("Object", (), kwargs)()

    return mock_pyjwk(key="mock-signing-key")


@pytest.fixture
def mock_inspections_list(mock_inspection):
    return [mock_inspection]


@pytest.fixture
def mock_token_str():
    return "mock-test-auth-token"


