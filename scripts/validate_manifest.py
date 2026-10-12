"""Compatibility wrapper for the package CLI."""

import sys

from kingwana_asr.cli import app

if __name__ == "__main__":
    app(["validate", *sys.argv[1:]])
