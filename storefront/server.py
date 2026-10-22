"""Marigold Coaches: the storefront of the fictional shop. It is a separate web app (python -m storefront.server, http://127.0.0.1:5000) that sits in front of the existing shop
(http://127.0.0.1:8802) and only calls the shop's own endpoints: GET /search, POST /book, POST /pay, POST /cancel. The shop's code, database logic, timings and telemetry are not changed.

  /                      home with a grid of popular journeys        /journeys?origin=&dest=   search results
  /journey/{id}          one journey                                 /cart, /checkout, /order/{id}
  /ops                   presenter controls (127.0.0.1 only): the only place where the fault names appear
  /healthz               for the launcher

Cart: a cookie that holds the journey details (no accounts, no personal data, no real payments)."""
import asyncio
import base64
import json
import os
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHOP_URL = os.environ.get("SHOP_URL", "http://127.0.0.1:8802")
BRAND = "Marigold Coaches"
CITIES = ["Berlin", "Paris", "Rome", "Madrid", "Vienna", "Prague", "Oslo", "Lisbon", "Athens", "Dublin", "Warsaw", "Zurich"]       # the shop's 12 cities (testbed/shop/db.py)
POPULAR = [("Berlin", "Prague"), ("Paris", "Madrid"), ("Rome", "Vienna"), ("Lisbon", "Madrid"), ("Zurich", "Paris"), ("Oslo", "Berlin")]
MAX_SEATS = 6

env = Environment(loader=FileSystemLoader(str(HERE / "templates")), autoescape=select_autoescape(["html"]))
env.filters["when"] = lambda ts: datetime.fromtimestamp(float(ts)).strftime("%a %d %b, %H:%M")
env.filters["money"] = lambda v: f"€{float(v):,.2f}"

ORDERS: dict[str, dict] = {}


class ShopDown(Exception):
    pass


async def shop(client: httpx.AsyncClient, method: str, path: str, **kw):
    """one call to the shop; any failure (shop down, 5xx, timeout) becomes ShopDown so that the page can say so in plain words"""
    try:
        r = await client.request(method, path, **kw)
    except (httpx.HTTPError, OSError) as e:
        raise ShopDown(repr(e))
    if r.status_code >= 500:
        raise ShopDown(f"shop answered {r.status_code}")
    return r


def render(request: Request, name: str, status: int = 200, **ctx) -> HTMLResponse:
    cart = read_cart(request)
    ctx.update(brand=BRAND, cart_count=sum(i["seats"] for i in cart), cities=CITIES, request=request)
    return HTMLResponse(env.get_template(name).render(**ctx), status_code=status)


def unavailable(request: Request) -> HTMLResponse:
    return render(request, "unavailable.html", status=503, title="Please try again")


# ------------------------------------------------------------------------------------------------ cart cookie
def read_cart(request: Request) -> list[dict]:
    raw = request.cookies.get("cart")
    if not raw:
        return []
    try:
        items = json.loads(base64.urlsafe_b64decode(raw.encode()).decode())
        return [dict(route_id=int(i["route_id"]), seats=max(1, min(MAX_SEATS, int(i["seats"]))), origin=str(i["origin"]), dest=str(i["dest"]), depart_ts=float(i["depart_ts"]), price=float(i["price"])) for i in items][:10]
    except Exception:
        return []


def write_cart(resp: Response, items: list[dict]):
    if not items:
        resp.delete_cookie("cart")
        return
    resp.set_cookie("cart", base64.urlsafe_b64encode(json.dumps(items).encode()).decode(), httponly=True, samesite="lax", max_age=3600 * 6)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        yield
        if _app.state.client is not None:
            await _app.state.client.aclose()

    app = FastAPI(title=BRAND, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    app.state.client = None

    def client() -> httpx.AsyncClient:
        if app.state.client is None:
            app.state.client = httpx.AsyncClient(base_url=SHOP_URL, timeout=8.0, limits=httpx.Limits(max_connections=40))
        return app.state.client

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "brand": BRAND}

    async def search(origin: str, dest: str):
        r = await shop(client(), "GET", "/search", params=dict(origin=origin, dest=dest))
        return r.json().get("results", []) if r.status_code == 200 else []

    # ------------------------------------------------------------------ browsing
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        try:
            found = await asyncio.gather(*[search(o, d) for o, d in POPULAR])
        except ShopDown:
            return unavailable(request)
        cards = []
        for (o, d), res in zip(POPULAR, found):
            if res:
                cards.append(dict(origin=o, dest=d, from_price=min(x["price"] for x in res), count=len(res), next_ts=res[0]["depart_ts"]))
        return render(request, "home.html", title="Coach journeys across Europe", cards=cards)

    @app.get("/journeys", response_class=HTMLResponse)
    async def journeys(request: Request, origin: str = "", dest: str = ""):
        if origin not in CITIES or dest not in CITIES or origin == dest:
            return render(request, "journeys.html", title="Find a journey", origin=origin, dest=dest, results=None, problem="Choose two different cities." if (origin or dest) else None)
        try:
            res = await search(origin, dest)
        except ShopDown:
            return unavailable(request)
        return render(request, "journeys.html", title=f"{origin} to {dest}", origin=origin, dest=dest, results=res, problem=None)

    @app.get("/journey/{route_id}", response_class=HTMLResponse)
    async def journey(request: Request, route_id: int, origin: str = "", dest: str = ""):
        if origin not in CITIES or dest not in CITIES:
            return render(request, "journeys.html", status=404, title="Journey not found", origin="", dest="", results=None, problem="We could not find that journey. Search again below.")
        try:
            res = await search(origin, dest)
        except ShopDown:
            return unavailable(request)
        hit = next((x for x in res if x["id"] == route_id), None)
        if hit is None:
            return render(request, "journeys.html", status=404, title="Journey not found", origin=origin, dest=dest, results=None, problem="That journey is no longer listed. Pick another departure.")
        return render(request, "journey.html", title=f"{origin} to {dest}", j=hit, others=[x for x in res if x["id"] != route_id][:4], max_seats=MAX_SEATS)

    # ------------------------------------------------------------------ cart and checkout
    @app.post("/cart/add")
    async def cart_add(request: Request, route_id: int = Form(...), seats: int = Form(1), origin: str = Form(...), dest: str = Form(...), depart_ts: float = Form(...), price: float = Form(...)):
        if origin not in CITIES or dest not in CITIES or not (1 <= seats <= MAX_SEATS) or price <= 0:
            return RedirectResponse("/", status_code=303)
        items = [i for i in read_cart(request) if i["route_id"] != route_id]
        items.append(dict(route_id=route_id, seats=seats, origin=origin, dest=dest, depart_ts=depart_ts, price=price))
        resp = RedirectResponse("/cart", status_code=303)
        write_cart(resp, items)
        return resp

    @app.post("/cart/remove")
    async def cart_remove(request: Request, route_id: int = Form(...)):
        resp = RedirectResponse("/cart", status_code=303)
        write_cart(resp, [i for i in read_cart(request) if i["route_id"] != route_id])
        return resp

    @app.get("/cart", response_class=HTMLResponse)
    async def cart(request: Request):
        items = read_cart(request)
        return render(request, "cart.html", title="Your cart", items=items, total=sum(i["price"] * i["seats"] for i in items))

    @app.get("/checkout", response_class=HTMLResponse)
    async def checkout_form(request: Request):
        items = read_cart(request)
        if not items:
            return RedirectResponse("/cart", status_code=303)
        return render(request, "checkout.html", title="Checkout", items=items, total=sum(i["price"] * i["seats"] for i in items), error=None, form_name="", form_email="")

    @app.post("/checkout", response_class=HTMLResponse)
    async def checkout(request: Request, name: str = Form(""), email: str = Form("")):
        items = read_cart(request)
        name, email = name.strip(), email.strip()
        total = sum(i["price"] * i["seats"] for i in items)
        if not items:
            return RedirectResponse("/cart", status_code=303)
        err = None
        if len(name) < 2:
            err = "Please enter the name for the booking."
        elif "@" not in email or "." not in email.split("@")[-1]:
            err = "Please enter an email address like name@example.com."
        if err:
            return render(request, "checkout.html", status=422, title="Checkout", items=items, total=total, error=err, form_name=name, form_email=email)
        booked = []
        try:
            for i in items:
                r = await shop(client(), "POST", "/book", json=dict(route_id=i["route_id"], user=name, seats=i["seats"]))
                if r.status_code != 200:
                    why = "is sold out" if r.status_code == 409 else "is no longer available"
                    return render(request, "checkout.html", status=409, title="Checkout", items=items, total=total, error=f"The {i['origin']} to {i['dest']} departure {why}. Remove it from your cart and pick another.", form_name=name, form_email=email)
                bid = r.json()["booking_id"]
                p = await shop(client(), "POST", "/pay", json=dict(booking_id=bid, amount=round(i["price"] * i["seats"], 2)))
                if p.status_code != 200:
                    return render(request, "checkout.html", status=409, title="Checkout", items=items, total=total, error="The payment step did not go through. Nothing was charged. Please try again.", form_name=name, form_email=email)
                booked.append(dict(i, booking_id=bid))
        except ShopDown:
            return unavailable(request)
        oid = secrets.token_hex(4).upper()
        ORDERS[oid] = dict(id=oid, name=name, email=email, items=booked, total=total, ts=time.time())
        if len(ORDERS) > 500:
            ORDERS.pop(next(iter(ORDERS)))
        resp = RedirectResponse(f"/order/{oid}", status_code=303)
        write_cart(resp, [])
        return resp

    @app.get("/order/{oid}", response_class=HTMLResponse)
    async def order(request: Request, oid: str):
        o = ORDERS.get(oid.upper())
        if o is None:
            return render(request, "journeys.html", status=404, title="Order not found", origin="", dest="", results=None, problem="We could not find that order.")
        return render(request, "order.html", title=f"Order {o['id']}", o=o)

    # ------------------------------------------------------------------ presenter controls (this computer only)
    from .ops import register as register_ops
    register_ops(app, render, shop_url=SHOP_URL)

    return app


app = create_app()


def main():
    import uvicorn
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host=os.environ.get("STOREFRONT_HOST", "127.0.0.1"), port=int(os.environ.get("STOREFRONT_PORT", "5000")), log_level="warning", loop="none", access_log=False)


if __name__ == "__main__":
    main()
