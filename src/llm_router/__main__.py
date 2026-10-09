"""``python -m llm_router``: serve on $HOST:$PORT (default 0.0.0.0:8080) using $ROUTER_CONFIG."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "llm_router.app:app_from_env",
        factory=True,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
