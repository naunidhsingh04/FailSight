"""Load generator for the ticket shop: an asyncio client with several traffic profiles, random user-behaviour mixes and random request rates, so that load changes
alone cannot be mistaken for faults.

  profiles: steady | bursty | ramp | daily      (rate(t) in requests per second; sessions arrive as an open-loop Poisson process)
  mixes:    random weights over four session types: browse (searches only), purchase (search, book, pay), abandon (search, book), refund (search, book, pay, cancel)

python -m testbed.loadgen --url http://127.0.0.1:8802 --duration 60 --profile bursty --rate 8 --seed 1
"""
import argparse
import asyncio
import math
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testbed.shop.db import CITIES  # noqa: E402

PROFILES = ("steady", "bursty", "ramp", "daily")
SESSIONS = ("browse", "purchase", "abandon", "refund")
AVG_REQUESTS = {"browse": 3.0, "purchase": 3.0, "abandon": 2.0, "refund": 4.0}


def make_rate_fn(profile, rate, duration, seed):
    rng = random.Random(seed)
    if profile == "steady":
        return lambda t: rate
    if profile == "ramp":
        lo, hi = rate * rng.uniform(0.3, 0.7), rate * rng.uniform(1.3, 1.8)
        return lambda t: lo + (hi - lo) * min(t / max(duration, 1), 1.0)
    if profile == "daily":
        period, phase, depth = rng.uniform(240, 600), rng.uniform(0, 2 * math.pi), rng.uniform(0.4, 0.7)
        return lambda t: max(rate * (1 + depth * math.sin(2 * math.pi * t / period + phase)), 0.2)
    if profile == "bursty":
        bursts, t = [], rng.uniform(20, 60)
        while t < duration:
            bursts.append((t, t + rng.uniform(8, 25), rng.uniform(2.5, 4.5)))
            t += rng.uniform(40, 120)
        return lambda t: rate * next((m for a, b, m in bursts if a <= t < b), 1.0)
    raise ValueError(profile)


def random_mix(seed):
    rng = random.Random(seed + 7)
    w = [rng.gammavariate(1.0, 1.0) + 0.05 for _ in SESSIONS]
    s = sum(w)
    return {k: v / s for k, v in zip(SESSIONS, w)}


class Stats:
    def __init__(self):
        self.requests = 0
        self.by_status = {}
        self.client_errors = 0
        self.sessions = 0
        self.shed = 0


class Shopper:
    def __init__(self, client, rng, stats):
        self.c, self.rng, self.s = client, rng, stats

    async def call(self, method, path, **kw):
        self.s.requests += 1
        for attempt in (0, 1):
            try:
                r = await self.c.request(method, path, **kw)
                self.s.by_status[r.status_code] = self.s.by_status.get(r.status_code, 0) + 1
                return r
            except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError) as e:       # the server closes the connection after an unhandled exception; retry once
                if attempt == 1:
                    self.s.client_errors += 1
                    return None
                await asyncio.sleep(0.05)
            except httpx.TimeoutException:
                self.s.client_errors += 1
                return None
        return None

    async def think(self):
        await asyncio.sleep(self.rng.uniform(0.05, 0.6))

    async def search(self):
        for _ in range(3):
            o, d = self.rng.sample(CITIES, 2)
            r = await self.call("GET", "/search", params=dict(origin=o, dest=d))
            if r is not None and r.status_code == 200 and r.json()["results"]:
                return self.rng.choice(r.json()["results"])
            if r is None or r.status_code >= 500:
                return None
        return None

    async def book(self, route):
        r = await self.call("POST", "/book", json=dict(route_id=route["id"], user=f"u{self.rng.randint(1, 5000)}", seats=self.rng.randint(1, 3)))
        return r.json()["booking_id"] if r is not None and r.status_code == 200 else None

    async def pay(self, bid, route):
        r = await self.call("POST", "/pay", json=dict(booking_id=bid, amount=route["price"]))
        return r is not None and r.status_code == 200

    async def session(self, kind):
        self.s.sessions += 1
        route = await self.search()
        if kind == "browse":
            if route is not None:
                await self.think()
                await self.search()
            return
        if route is None:
            return
        await self.think()
        bid = await self.book(route)
        if bid is None or kind == "abandon":
            return
        await self.think()
        ok = await self.pay(bid, route)
        if kind == "refund" and ok:
            await self.think()
            await self.call("POST", "/cancel", json=dict(booking_id=bid))


async def run(url, duration, profile="steady", rate=6.0, seed=1, mix=None, max_sessions=300, stop=None, log=None):
    rng = random.Random(seed)
    mix = mix or random_mix(seed)
    rate_fn = make_rate_fn(profile, rate, duration, seed)
    stats = Stats()
    kinds, weights = list(mix), [mix[k] for k in mix]
    avg_req = sum(mix[k] * AVG_REQUESTS[k] for k in mix)
    tasks = set()
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
    async with httpx.AsyncClient(base_url=url, timeout=6.0, limits=limits) as client:
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= duration or (stop is not None and stop.is_set()):
                break
            if len(tasks) >= max_sessions:
                stats.shed += 1
            else:
                kind = rng.choices(kinds, weights)[0]
                task = asyncio.create_task(Shopper(client, random.Random(rng.random()), stats).session(kind))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            lam = max(rate_fn(t) / avg_req, 0.05)
            await asyncio.sleep(rng.expovariate(lam))
        if tasks:
            await asyncio.wait(tasks, timeout=8)
    if log:
        log(f"loadgen done: {stats.requests} requests, status {dict(sorted(stats.by_status.items()))}, client errors {stats.client_errors}, sessions {stats.sessions}, shed {stats.shed}")
    return stats, dict(profile=profile, rate=rate, mix=mix, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8802")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--profile", choices=PROFILES, default="steady")
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(a.url, a.duration, a.profile, a.rate, a.seed, log=print))


if __name__ == "__main__":
    main()
