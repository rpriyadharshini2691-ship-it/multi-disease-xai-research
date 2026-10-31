"""Launch the local API or Streamlit app with isolated SQLite development settings."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def configure_environment(api_port: int) -> None:
    os.environ["DATABASE_URL"] = "sqlite:///./xai_dashboard.db"
    os.environ["ENVIRONMENT"] = "development"
    os.environ.setdefault(
        "SECRET_KEY", "local-development-secret-change-before-deployment"
    )
    os.environ["PYTHONPATH"] = str(SRC)
    os.environ["API_BASE_URL"] = f"http://127.0.0.1:{api_port}/api/v1"
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("api", "ui"))
    parser.add_argument("--api-port", type=int, default=8002)
    parser.add_argument("--ui-port", type=int, default=8502)
    args = parser.parse_args()
    configure_environment(args.api_port)
    if args.service == "api":
        import uvicorn

        uvicorn.run("xai_dashboard.api.main:app", host="127.0.0.1", port=args.api_port)
        return
    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        str(SRC / "xai_dashboard" / "ui" / "Home.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(args.ui_port),
    ]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
