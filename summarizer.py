"""Compatibility wrapper named `summarizer` to satisfy notebook imports.

The original repo included a vendored `summarizer` package. To avoid
modifying many notebook files we provide a tiny wrapper that forwards
the `summarize` function to our `summarizer_custom.GetText` implementation.
"""
from typing import Optional
import summarizer_custom as _sc


def summarize(title: str, text: str, count: int = 3, summarizer: Optional[object] = None):
    """Return a short summary string.

    Keeps a similar signature to the upstream `summarizer.summarize` so
    notebooks that import it continue to work.
    """
    try:
        return _sc.GetText(title, text, count)
    except Exception:
        # conservative fallback
        return text if text else ''


__all__ = ["summarize"]
