"""Small SQLite data layer with a real connection pool and real queries for the ticket shop."""
import queue
import random
import sqlite3
import threading
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS routes (id INTEGER PRIMARY KEY, origin TEXT, dest TEXT, depart_ts INTEGER, price REAL, seats INTEGER);
CREATE INDEX IF NOT EXISTS ix_routes_od ON routes(origin, dest);
CREATE TABLE IF NOT EXISTS bookings (id INTEGER PRIMARY KEY AUTOINCREMENT, route_id INTEGER, user TEXT, seats INTEGER, status TEXT, created_ts REAL);
CREATE INDEX IF NOT EXISTS ix_bookings_route ON bookings(route_id);
CREATE TABLE IF NOT EXISTS payments (id INTEGER PRIMARY KEY AUTOINCREMENT, booking_id INTEGER, amount REAL, ts REAL);
"""
CITIES = ["Berlin", "Paris", "Rome", "Madrid", "Vienna", "Prague", "Oslo", "Lisbon", "Athens", "Dublin", "Warsaw", "Zurich"]


class PoolTimeout(Exception):
    pass


class Pool:
    """fixed-size connection pool: acquire() waits for a free connection (up to `timeout` s) and returns it afterwards"""

    def __init__(self, path, size=8, timeout=3.0, busy_timeout=3.0):
        self.path, self.size, self.timeout = path, size, timeout
        self.q = queue.Queue()
        self.leaked = []                                  # connections held by the pool-exhaustion fault
        self._lock = threading.Lock()
        for _ in range(size):
            c = sqlite3.connect(path, check_same_thread=False, timeout=busy_timeout, isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.row_factory = sqlite3.Row
            self.q.put(c)

    @contextmanager
    def acquire(self):
        try:
            conn = self.q.get(timeout=self.timeout)
        except queue.Empty:
            raise PoolTimeout(f"no free database connection after {self.timeout:.1f}s")
        try:
            yield conn
        finally:
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            self.q.put(conn)

    def free(self):
        return self.q.qsize()

    def hold_one(self):
        """take one connection out of circulation (used by the pool-exhaustion fault); returns False when none is free"""
        try:
            c = self.q.get_nowait()
        except queue.Empty:
            return False
        self.leaked.append(c)
        return True

    def release_held(self):
        while self.leaked:
            self.q.put(self.leaked.pop())


def init_db(path, seed=1, n_routes=400):
    c = sqlite3.connect(path, isolation_level=None)
    c.executescript(SCHEMA)
    if c.execute("SELECT count(*) FROM routes").fetchone()[0] == 0:
        rng = random.Random(seed)
        now = int(time.time())
        c.execute("BEGIN")
        for i in range(n_routes):
            o, d = rng.sample(CITIES, 2)
            c.execute("INSERT INTO routes(origin,dest,depart_ts,price,seats) VALUES (?,?,?,?,?)", (o, d, now + rng.randint(3600, 86400 * 30), round(rng.uniform(19, 249), 2), rng.randint(20, 120)))
        c.execute("COMMIT")
    c.close()


# ---------------------------------------------------------------------------------------------------- queries
def search(conn, origin, dest, limit=20):
    return [dict(r) for r in conn.execute(
        "SELECT r.id, r.origin, r.dest, r.depart_ts, r.price, r.seats - COALESCE(SUM(CASE WHEN b.status != 'cancelled' THEN b.seats END), 0) AS free "
        "FROM routes r LEFT JOIN bookings b ON b.route_id = r.id WHERE r.origin = ? AND r.dest = ? GROUP BY r.id ORDER BY r.depart_ts LIMIT ?", (origin, dest, limit))]


def book(conn, route_id, user, seats):
    conn.execute("BEGIN IMMEDIATE")
    r = conn.execute("SELECT seats FROM routes WHERE id = ?", (route_id,)).fetchone()
    if r is None:
        conn.execute("ROLLBACK")
        return None, "no such route"
    taken = conn.execute("SELECT COALESCE(SUM(seats),0) FROM bookings WHERE route_id = ? AND status != 'cancelled'", (route_id,)).fetchone()[0]
    if taken + seats > r["seats"]:
        conn.execute("ROLLBACK")
        return None, "sold out"
    cur = conn.execute("INSERT INTO bookings(route_id,user,seats,status,created_ts) VALUES (?,?,?,?,?)", (route_id, user, seats, "reserved", time.time()))
    conn.execute("COMMIT")
    return cur.lastrowid, None


def pay(conn, booking_id, amount):
    conn.execute("BEGIN IMMEDIATE")
    b = conn.execute("SELECT status FROM bookings WHERE id = ?", (booking_id,)).fetchone()
    if b is None or b["status"] != "reserved":
        conn.execute("ROLLBACK")
        return False, "booking not payable"
    conn.execute("UPDATE bookings SET status = 'paid' WHERE id = ?", (booking_id,))
    conn.execute("INSERT INTO payments(booking_id,amount,ts) VALUES (?,?,?)", (booking_id, amount, time.time()))
    conn.execute("COMMIT")
    return True, None


def cancel(conn, booking_id):
    conn.execute("BEGIN IMMEDIATE")
    b = conn.execute("SELECT status FROM bookings WHERE id = ?", (booking_id,)).fetchone()
    if b is None or b["status"] == "cancelled":
        conn.execute("ROLLBACK")
        return False
    conn.execute("UPDATE bookings SET status = 'cancelled' WHERE id = ?", (booking_id,))
    conn.execute("COMMIT")
    return True


def heavy_query(conn, target_ms):
    """real database work (a CPU-bound self-join), repeated until about target_ms have passed: the 'slow query' fault"""
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000 < target_ms:
        conn.execute("SELECT count(*) FROM routes a, routes b WHERE (a.id * b.id + a.seats) % 7 = 0 AND a.price < b.price").fetchone()
