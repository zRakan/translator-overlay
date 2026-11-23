import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from translator_overlay import translate


class Server:
    """Local HTTP/1.1 server that counts connections and can misbehave on request."""

    def __init__(self):
        outer = self
        self.connections, self.requests, self.mode = 0, [], "ok"

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                outer.connections += 1

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                mode = outer.mode  # snapshot: the test flips it as soon as the client has its answer
                status = 500 if mode == "error" else 200
                payload = json.dumps({"choices": [{"text": " مرحبا "}], "n": len(outer.requests)} if status == 200 else {"error": "x"}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                if mode == "close":
                    self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                if mode == "drop":  # silently hang up after answering, as an idle keep-alive timeout does
                    self.close_connection = True

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/completions"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.fixture
def conn():
    return translate._Connection()


HDR = {"Content-Type": "application/json"}


def test_one_connection_serves_consecutive_requests(server, conn):
    for i in range(3):
        assert conn.post_json(server.url, {"i": i}, HDR)["n"] == i + 1
    assert server.connections == 1 and [b["i"] for _, b in server.requests] == [0, 1, 2]


def test_reconnects_after_the_server_announces_close(server, conn):
    server.mode = "close"
    conn.post_json(server.url, {}, HDR)
    server.mode = "ok"
    assert conn.post_json(server.url, {}, HDR)["n"] == 2 and server.connections == 2


def test_stale_connection_is_retried_once_transparently(server, conn):
    server.mode = "drop"
    conn.post_json(server.url, {"a": 1}, HDR)
    server.mode = "ok"
    assert conn.post_json(server.url, {"a": 2}, HDR)["n"] == 2  # first attempt hits the closed socket, the retry succeeds
    assert [b["a"] for _, b in server.requests] == [1, 2] and server.connections == 2


def test_error_status_raises_and_the_next_request_still_works(server, conn):
    server.mode = "error"
    with pytest.raises(RuntimeError, match="HTTP 500"):
        conn.post_json(server.url, {}, HDR)
    server.mode = "ok"
    assert conn.post_json(server.url, {}, HDR)["choices"][0]["text"].strip() == "مرحبا"


def test_unreachable_server_raises_instead_of_retrying_forever(conn):
    with pytest.raises(OSError):
        conn.post_json("http://127.0.0.1:9/v1/completions", {}, HDR, timeout=1)


def test_chat_backend_end_to_end(server, monkeypatch):
    monkeypatch.setattr(translate, "_connection", translate._Connection())
    chat = translate.Chat(server.url.replace("/completions", "/chat/completions"), "", "ar", style="plamo-pfml")
    assert chat.translate("Hello there.") == "مرحبا"
    assert chat.translate("Again.") == "مرحبا" and server.connections == 1
    path, body = server.requests[0]
    assert path == "/v1/completions" and body["cache_prompt"] is True and body["temperature"] == 0 and "Hello there." in body["prompt"]
