"""Console entrypoint for ``nre-serve`` and ``run_server.py``."""

from __future__ import annotations

import argparse
import logging
import sys

from nre.primitives import ensure_primitive_logging_visible

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = "info"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Kernel primitives: nre.kernel, nre.decompose, …; API / OpenRouter: nre.api, nre.openrouter, …
    logging.getLogger("nre").setLevel(getattr(logging, level.upper(), logging.INFO))
    ensure_primitive_logging_visible()


def main() -> None:
    parser = argparse.ArgumentParser(description="NRE kernel FastAPI server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload (dev; single worker)",
    )
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Uvicorn log level")
    args = parser.parse_args()

    _configure_logging(args.log_level)

    import uvicorn

    print("=" * 72)
    print("NRE KERNEL SERVER")
    print("=" * 72)
    print(f"URL:     http://{args.host}:{args.port}")
    print(f"Health:  http://{args.host}:{args.port}/health")
    print(
        f"Chat:    POST http://{args.host}:{args.port}/v1/chat/completions "
        "(kernel trace -> base OpenRouter -> tool_calls)",
    )
    print(f"Models:  GET http://{args.host}:{args.port}/v1/models")
    print("=" * 72)

    uvicorn.run(
        "nre.server.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
