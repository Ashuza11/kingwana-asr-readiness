#!/usr/bin/env python3
"""Serve the private manual-review interface on localhost only."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from kingwana_asr.data.manifest import ManifestError  # noqa: E402
from kingwana_asr.evaluation.manual_review_web import (  # noqa: E402
    REVIEW_LABELS,
    ManualReviewStore,
)

ASSET_DIRECTORY = REPOSITORY_ROOT / "tools" / "manual_review"
DEFAULT_REVIEW = REPOSITORY_ROOT / "reports" / "manual_review" / "manual_error_review.csv"
DEFAULT_PROGRESS = (
    REPOSITORY_ROOT / "reports" / "manual_review" / "manual_error_review_progress.json"
)
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ReviewHandler(BaseHTTPRequestHandler):
    """Small same-origin API for local review decisions."""

    store: ManualReviewStore

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/review":
            try:
                rows, progress = self.store.load()
                self._json(
                    HTTPStatus.OK,
                    {"rows": rows, "progress": progress, "allowed_labels": REVIEW_LABELS},
                )
            except ManifestError as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            try:
                body = (ASSET_DIRECTORY / filename).read_bytes()
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/decisions":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ManifestError("Request body is empty or too large")
            payload = json.loads(self.rfile.read(length))
            decisions = payload.get("decisions") if isinstance(payload, dict) else None
            if not isinstance(decisions, list):
                raise ManifestError("Request must contain a decisions list")
            progress = self.store.apply(decisions)
            self._json(HTTPStatus.OK, {"ok": True, "progress": progress})
        except (ManifestError, json.JSONDecodeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"manual-review: {format_string % args}")

    def _json(self, status: HTTPStatus, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ReviewHandler.store = ManualReviewStore(args.review.resolve(), args.progress.resolve())
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Manual review is available at {url}")
    print("Decisions are saved locally after each action. Press Ctrl+C to stop.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping manual-review server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
