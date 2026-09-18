"""Run the jev-local server: `python -m jevlocal [--port 8000]`."""

import argparse
import os


def serve(port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("jevlocal.app:app", host="0.0.0.0", port=port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="jev-local-serve")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    serve(args.port)


if __name__ == "__main__":
    main()
