from quack.core import __VERSION__, get_version

#: Canonical version string, kept in sync with the distribution metadata
#: (`pyproject.toml` reads it from `quack.core.__VERSION__`).
__version__ = __VERSION__

__all__ = ['get_version', '__version__']
