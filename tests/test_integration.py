"""Integration pieces: the nginx log tailer, the Node/Express middleware, the optional per-app API key check and the Slack-compatible alert webhook."""
import json
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
from failsight_agent import tail  # noqa: E402

CFG = ROOT / "report_outputs" / "testbed_frozen_config.json"
MODEL = ROOT / "models" / "testbed" / "detector.joblib"
RUN = "r004_s1004"


class Capture(BaseHTTPRequestHandler):
    bodies = []

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        Capture.bodies.append((self.path, dict(self.headers), json.loads(self.rfile.read(n) or b"{}")))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    Capture.bodies = []
    srv = HTTPServer(("127.0.0.1", 0), Capture)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_nginx_combined_and_request_time_lines_are_parsed():
    a = tail.parse_line('203.0.113.9 - - [21/Oct/2026:13:55:01 +0000] "GET /search?from=A HTTP/1.1" 200 1532 "-" "curl/8.0"')
    assert a["method"] == "GET" and a["route"] == "/search" and a["status_code"] == 200 and a["content_length"] == 1532 and a["latency_ms"] == 0.0
    assert a["timestamp"] == pytest.approx(1792590901.0)
    b = tail.parse_line('203.0.113.9 - - [21/Oct/2026:13:55:02 +0000] "POST /pay HTTP/2.0" 502 - "-" "x" 0.250')
    assert b["status_code"] == 502 and b["content_length"] == 0 and b["latency_ms"] == pytest.approx(250.0)
    assert tail.parse_line("not an access log line") is None


def test_tailer_sends_new_lines_to_the_monitor(tmp_path, server):
    log = tmp_path / "access.log"
    log.write_text("", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-m", "failsight_agent.tail", "--file", str(log), "--monitor", server, "--app-id", "site", "--api-key", "k1"], cwd=ROOT / "agent")
    time.sleep(1.5)
    with open(log, "a", encoding="utf-8") as f:
        for i in range(3):
            f.write(f'1.2.3.4 - - [21/Oct/2026:13:55:0{i} +0000] "GET /x HTTP/1.1" 200 10 "-" "t"\n')
        f.write("garbage\n")
    time.sleep(3)
    p.terminate()
    p.wait(timeout=10)
    got = [e for path, h, b in Capture.bodies if path == "/ingest/api" for e in b["events"]]
    assert len(got) == 3 and all(b["system"] == "site" for _, _, b in Capture.bodies)
    assert {k.lower(): v for k, v in Capture.bodies[0][1].items()}.get("x-api-key") == "k1"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_express_middleware_posts_the_same_fields(server):
    js = f"""
    const failsight = require({json.dumps(str(ROOT / 'agent' / 'node' / 'failsight-express.js'))});
    const {{ EventEmitter }} = require('events');
    const mw = failsight({{ monitorUrl: {json.dumps(server)}, appId: 'node-app', apiKey: 'sekret', flushMs: 100000 }});
    const res = new EventEmitter(); res.statusCode = 503; res.get = () => '42';
    mw({{ method: 'GET', path: '/pay', url: '/pay' }}, res, () => {{}});
    res.emit('finish');
    mw.flush().then(() => process.exit(0));
    """
    subprocess.run(["node", "-e", js], check=True, timeout=30)
    path, headers, body = Capture.bodies[0]
    ev = body["events"][0]
    assert path == "/ingest/api" and body["system"] == "node-app" and {k.lower(): v for k, v in headers.items()}.get("x-api-key") == "sekret"
    assert set(ev) >= {"timestamp", "method", "route", "status_code", "latency_ms", "content_length"} and ev["status_code"] == 503 and ev["route"] == "/pay" and ev["content_length"] == 42


def test_api_key_check_is_off_by_default_and_per_app():
    from app.backend.testbed_monitor import check_api_key, parse_keys
    assert parse_keys(None) == {} and parse_keys("") == {}
    assert check_api_key("shop", None, {}) is True                                   # off by default
    keys = parse_keys("shop:abc, other:xyz")
    assert keys == {"shop": "abc", "other": "xyz"}
    assert check_api_key("shop", "abc", keys) and not check_api_key("shop", "xyz", keys) and not check_api_key("shop", None, keys) and not check_api_key("unknown", "abc", keys)


@pytest.mark.skipif(not (CFG.exists() and MODEL.exists() and (ROOT / "testbed" / "runs" / RUN / "DONE").exists()), reason="frozen detector or run not present")
def test_alert_webhook_receives_a_slack_compatible_message(server):
    from app.backend import testbed_monitor as TM
    from testbed import dataset as DS
    TM.LIVE_DIR = ROOT / "outputs" / "testbed_live_test"
    mon = TM.TestbedMonitor()
    mon.set_webhook(server + "/hook")
    d = DS.RUNS / RUN
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    mon.reset("hooked", meta["start_ts"])
    for kind in ("api", "logs", "metrics"):
        ev = [json.loads(line) for line in open(d / f"{kind}.jsonl", encoding="utf-8")]
        for i in range(0, len(ev), 2000):
            mon.ingest(kind, "hooked", ev[i:i + 2000])
    mon.close_ready(meta["start_ts"] + 100000)
    time.sleep(2)
    hooks = [b for path, _, b in Capture.bodies if path == "/hook"]
    assert len(hooks) >= 1 and all(set(h) == {"text"} for h in hooks)
    assert "FailSight alert for hooked" in hooks[0]["text"] and "failure risk" in hooks[0]["text"]
    assert len(hooks) == sum(1 for w in mon.windows("hooked") if w["new_alert"])           # one message per alert episode
    mon.set_webhook(None)
    assert mon.webhook_url is None
