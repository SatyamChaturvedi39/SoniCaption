from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_labs():
    assert client.post("/api/v1/labs/dtw").status_code == 200
    assert client.post("/api/v1/labs/hmm/viterbi").status_code == 200

def test_invalid_upload():
    r = client.post("/api/v1/audio/analyze", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert r.status_code == 415
