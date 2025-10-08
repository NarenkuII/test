"""signlang package initialization."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("signlang")
except PackageNotFoundError:  # pragma: no cover - during local development
    __version__ = "0.0.0"
