from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for JSON logs and metadata."""

    return datetime.now(UTC)