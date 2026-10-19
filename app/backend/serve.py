"""Launcher: python -m app.backend.serve   (env HOST, PORT). On Windows it selects the selector event loop, because uvicorn's default Proactor loop
can stop accepting connections ("Accept failed on a socket") when browsers (notably WebKit) reset speculative connections."""
import asyncio
import os
import sys

import uvicorn


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("app.backend.main:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), log_level=os.environ.get("LOG_LEVEL", "warning"), loop="none")


if __name__ == "__main__":
    main()
