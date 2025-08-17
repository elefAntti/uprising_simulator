# bots/__init__.py
from __future__ import annotations
import importlib, pkgutil

# Global registry: name -> class
BOT_REGISTRY = {}

def register_bot(cls=None, *, name: str | None = None):
    """Decorator to register a bot class under a human-friendly name."""
    def _wrap(c):
        BOT_REGISTRY[name or c.__name__] = c
        return c
    return _wrap(cls) if cls is not None else _wrap

def load_all_bots():
    """Import every module in bots/ so decorators run and fill BOT_REGISTRY."""
    import bots as _pkg
    for _, modname, ispkg in pkgutil.iter_modules(_pkg.__path__):
        if ispkg or modname.startswith("_"):
            continue
        importlib.import_module(f"bots.{modname}")

def get_bot_registry() -> dict[str, type]:
    return dict(BOT_REGISTRY)


keyboard_listeners=[]
