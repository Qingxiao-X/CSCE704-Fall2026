"""End-to-end tests for src/server.py against the checked-in model:
every POST must get a JSON {"result": 0|1}, whatever the body."""

import http.client
import json
import os
import threading
from http.server import ThreadingHTTPServer

import pytest

import src.server as server

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
VALID_PE = os.path.join(FIXTURES, "valid_sample.exe")


@pytest.fixture(scope="module")
def port():
    server.DETECTOR = server.Detector(server.MODEL_DIR)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def _post(port, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/", body=body, headers={"Content-Type": "application/octet-stream"})
    resp = conn.getresponse()
    return resp.status, resp.getheader("Content-Type"), json.loads(resp.read())


def test_ready_probe(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/")
    assert conn.getresponse().status == 200


def test_benign_fixture_is_goodware(port):
    with open(VALID_PE, "rb") as f:
        assert _post(port, f.read()) == (200, "application/json", {"result": 0})


@pytest.mark.parametrize("body", [b"", b"not a pe", b"MZ", b"MZ" + os.urandom(5000)])
def test_any_body_gets_integer_result(port, body):
    status, ctype, payload = _post(port, body)
    assert status == 200 and ctype == "application/json"
    assert payload["result"] in (0, 1) and type(payload["result"]) is int


def test_unscorable_input_gets_fallback(port):
    assert _post(port, b"")[2] == {"result": server.FALLBACK_RESULT}
