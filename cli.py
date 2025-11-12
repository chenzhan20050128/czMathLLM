"""Compatibility shim for importing the CLI entry point."""

from __future__ import annotations

from .cli_core import main  # re-export for `python cli.py`
