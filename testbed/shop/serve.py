"""python -m testbed.shop.serve   (env: PORT, SHOP_DB, SHOP_EVENTS, FAILSIGHT_URL, FAILSIGHT_KEY, SHOP_SEED). Windows: selector event loop (as app/backend/serve.py)."""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))                     # the agent is a separate package: it is imported exactly as a third-party app would import it

import uvicorn  # noqa: E402


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("testbed.shop.app:app_from_env", factory=True, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8802")), log_level="warning", loop="none", access_log=False)


if __name__ == "__main__":
    main()
