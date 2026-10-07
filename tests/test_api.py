"""API smoke tests.

Deliberately restricted to request paths that perform **no outbound network
access** (whitelisted domain / malformed body). Network-dependent behaviour is
covered by the extraction tests in milestone P2.
"""

import pytest

import app as app_module


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def test_root_reports_running(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.get_json()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model"] == "random_forest.joblib"


def test_predict_whitelisted_domain_skips_model_and_returns_safe(client):
    response = client.post("/predict", json={"url": "https://www.google.com/search?q=x"})
    assert response.status_code == 200
    assert response.get_json() == {"prediction": 0}


def test_predict_trusted_subdomain_is_whitelisted(client):
    response = client.post("/predict", json={"url": "https://mail.google.com/mail"})
    assert response.status_code == 200
    assert response.get_json() == {"prediction": 0}


def test_predict_empty_body_is_a_structured_client_error(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    body = response.get_json()
    assert "error" in body
    assert "Traceback" not in body["error"]
