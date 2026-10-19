"""ASGI middleware: records time, route, method, status and latency of every HTTP request (FastAPI / Starlette / any ASGI app)."""
import asyncio
import time


class FailSightMiddleware:
    def __init__(self, app, agent):
        self.app = app
        self.agent = agent

    async def __call__(self, scope, receive, send):
        if not self.agent._loop_started:
            self.agent._start_loop_monitor()
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t0 = time.perf_counter()
        wall = time.time()
        state = {"status": 500, "length": 0}
        self.agent._inflight += 1

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                for k, v in message.get("headers", []):
                    if k == b"content-length":
                        try:
                            state["length"] = int(v)
                        except ValueError:
                            pass
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            state["status"] = 500
            raise
        finally:
            self.agent._inflight -= 1
            route = scope.get("route")
            path = getattr(route, "path", None) or scope.get("path", "")
            self.agent.client.emit("api", dict(timestamp=wall, method=scope.get("method", ""), route=path, status_code=state["status"],
                                              latency_ms=round((time.perf_counter() - t0) * 1000.0, 3), content_length=state["length"]))
