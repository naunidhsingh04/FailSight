"""Tail an nginx (or any combined-format) access log and send each request to a FailSight monitor.

    python -m failsight_agent.tail --file /var/log/nginx/access.log --monitor http://127.0.0.1:8000 --app-id mysite [--api-key KEY] [--from-start]

Understands the nginx / Apache "combined" format; if the line ends with a request time in seconds (log_format ... '$request_time') it is used as the latency, otherwise the latency
is sent as 0 and the monitor can only use counts and error rates. Standard library only."""
import argparse
import json
import re
import sys
import time
from datetime import datetime

from .client import Client

LINE = re.compile(r'^(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>[A-Z]+) (?P<path>\S+)[^"]*" (?P<status>\d{3}) (?P<bytes>\d+|-)(?: "[^"]*" "[^"]*")?(?: (?P<rt>\d+\.\d+))?\s*$')


def parse_line(line):
    """one access-log line -> an event dict for /ingest/api, or None when the line is not an access-log line"""
    m = LINE.match(line.strip())
    if not m:
        return None
    try:
        ts = datetime.strptime(m.group("time"), "%d/%b/%Y:%H:%M:%S %z").timestamp()
    except ValueError:
        return None
    path = m.group("path").split("?")[0]
    return dict(timestamp=ts, method=m.group("method"), route=path, status_code=int(m.group("status")), latency_ms=float(m.group("rt")) * 1000.0 if m.group("rt") else 0.0,
                content_length=0 if m.group("bytes") == "-" else int(m.group("bytes")))


def follow(path, from_start=False, poll=0.5, stop=lambda: False):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if not from_start:
            f.seek(0, 2)
        while not stop():
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(poll)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--monitor", required=True)
    ap.add_argument("--app-id", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--from-start", action="store_true", help="also send the lines that are already in the file")
    a = ap.parse_args(argv)
    c = Client(a.app_id, a.monitor, a.api_key).start()
    sent = skipped = 0
    try:
        for line in follow(a.file, a.from_start):
            ev = parse_line(line)
            if ev is None:
                skipped += 1
                continue
            c.emit("api", ev)
            sent += 1
    except KeyboardInterrupt:
        pass
    finally:
        c.stop()
        print(json.dumps(dict(sent=sent, skipped=skipped)), file=sys.stderr)


if __name__ == "__main__":
    main()
