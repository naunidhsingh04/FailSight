"""The ticket shop: a small real FastAPI + SQLite app with 5 endpoints (search, book, pay, cancel, health), structured logging and a connection pool.

It is monitored ONLY through the failsight_agent package (the three lines marked 'FailSight' below). There is no other reporting code in this app; the /admin
endpoints belong to the fault injector (they change the app's behaviour, they report nothing)."""
import logging
import os
import random
import sqlite3
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import db
from .faults import FaultRegistry

log = logging.getLogger("shop")


class BookIn(BaseModel):
    route_id: int
    user: str
    seats: int = 1


class PayIn(BaseModel):
    booking_id: int
    amount: float


class CancelIn(BaseModel):
    booking_id: int


class FaultIn(BaseModel):
    type: str
    duration_s: float = 60.0
    params: dict = {}


def create_app(db_path, events_path=None, failsight=None, seed=1, pool_size=8):
    db.init_db(db_path, seed=seed)
    pool = db.Pool(db_path, size=pool_size)
    faults = FaultRegistry(events_path, pool)
    app = FastAPI(title="shop", docs_url=None, redoc_url=None, openapi_url=None)
    if failsight is not None:
        failsight.instrument(app, pool=pool)                                  # FailSight: request middleware, log forwarding, process sampler, pool wait

    def guarded(route, fn):
        """run a request body on the pool: faults first, then the real queries"""
        try:
            faults.before_request(route)
            with pool.acquire() as conn:
                faults.db_extra(conn)
                return fn(conn)
        except db.PoolTimeout as e:
            log.error("%s: %s", route, e)
            raise HTTPException(503, "database busy")
        except TimeoutError as e:
            log.error("%s: %s", route, e)
            raise HTTPException(504, "timeout")
        except sqlite3.OperationalError as e:
            log.error("%s: database error: %s", route, e)
            raise HTTPException(500, "database error")
        except MemoryError as e:
            log.error("%s: %s", route, e)
            raise HTTPException(500, "out of memory")
        except HTTPException:
            raise
        except RuntimeError as e:
            log.error("%s: %s", route, e)
            raise HTTPException(500, "internal error")

    @app.get("/health")
    def health():
        return {"status": "ok", "pool_free": pool.free(), "faults_active": len(faults.status())}

    @app.get("/search")
    def search(origin: str, dest: str):
        res = guarded("search", lambda c: db.search(c, origin, dest))
        log.info("search %s-%s: %d results", origin, dest, len(res))
        return {"results": res}

    @app.post("/book")
    def book(b: BookIn):
        bid, err = guarded("book", lambda c: db.book(c, b.route_id, b.user, b.seats))
        if err:
            log.warning("book refused: %s", err)
            raise HTTPException(409 if err == "sold out" else 404, err)
        log.info("booked %s route=%s seats=%s", bid, b.route_id, b.seats)
        return {"booking_id": bid}

    @app.post("/pay")
    def pay(p: PayIn):
        time.sleep(random.uniform(0.005, 0.025))                              # the payment gateway
        ok, err = guarded("pay", lambda c: db.pay(c, p.booking_id, p.amount))
        if not ok:
            log.warning("payment refused: %s", err)
            raise HTTPException(409, err)
        log.info("paid booking %s amount=%.2f", p.booking_id, p.amount)
        return {"paid": True}

    @app.post("/cancel")
    def cancel(c_: CancelIn):
        ok = guarded("cancel", lambda c: db.cancel(c, c_.booking_id))
        if not ok:
            raise HTTPException(404, "no such booking")
        log.info("cancelled booking %s", c_.booking_id)
        return {"cancelled": True}

    # ---- fault injector (testbed only): changes behaviour, reports nothing
    @app.post("/admin/fault")
    def admin_fault(f: FaultIn):
        try:
            return {"fault_id": faults.start(f.type, f.duration_s, f.params)}
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/admin/faults")
    def admin_faults():
        return {"active": faults.status()}

    @app.post("/admin/clear")
    def admin_clear():
        faults.clear()
        return {"cleared": True}

    app.state.pool, app.state.faults = pool, faults
    return app


def app_from_env():
    """uvicorn factory: configuration through environment variables (set by the run generator / launcher)"""
    logging.basicConfig(level=logging.INFO, format='{"ts": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": "%(message)s"}')
    fs = None
    if os.environ.get("FAILSIGHT_URL"):
        from failsight_agent import FailSight
        fs = FailSight.from_config(dict(app_id=os.environ.get("SHOP_ID", "shop"), monitor_url=os.environ["FAILSIGHT_URL"], api_key=os.environ.get("FAILSIGHT_KEY")))
    return create_app(os.environ.get("SHOP_DB", "shop.db"), os.environ.get("SHOP_EVENTS"), fs, int(os.environ.get("SHOP_SEED", "1")))
