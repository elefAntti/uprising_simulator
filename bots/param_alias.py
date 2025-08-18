
"""
bots/param_alias.py
-------------------
Register alias bot classes that bake in parameter genomes (e.g., from best_params.json).
Lets you refer to them by name in scripts without changing those scripts' logic.

Usage
-----
import bots.param_alias as PA
PA.register_from_json("ConeKeeper", "best_params.json", alias="best")  # -> "ConeKeeper@best"
# or auto-load several:
PA.autoload("zoo/**/*.json")  # registers e.g. "ConeKeeper@gen12_abcd1234"

Then in your code:
  from bots import load_all_bots, get_bot_registry
  load_all_bots()
  REG = get_bot_registry()
  bot = REG["ConeKeeper@best"](0)  # params baked in
"""
from __future__ import annotations
import json, glob, os, re, hashlib
from typing import Dict, Any, Optional

from bots import register_bot, get_bot_registry

def _extract_genome(obj: Dict[str, Any]) -> Dict[str, float]:
    if "genome" in obj and isinstance(obj["genome"], dict):
        return obj["genome"]
    # Allow direct dict of params
    return {k: v for k, v in obj.items() if isinstance(v, (int, float))}

def _hash_params(params: Dict[str, float]) -> str:
    s = ",".join(f"{k}={params[k]:.6g}" for k in sorted(params.keys()))
    return hashlib.sha1(s.encode()).hexdigest()[:8]

def register_alias(base_name: str, params: Dict[str, float], alias: Optional[str] = None) -> str:
    REG = get_bot_registry()
    if base_name not in REG:
        raise KeyError(f"Unknown base bot: {base_name}")
    Base = REG[base_name]
    params = dict(params)
    tag = alias or _hash_params(params)
    name = f"{base_name}@{tag}"

    # Create a thin wrapper class that forwards to Base with baked params
    @register_bot(name=name)
    class _Alias(Base):  # type: ignore[misc]
        __doc__ = f"{base_name} with baked params alias='{tag}'"
        def __init__(self, index: int, **kw):
            merged = dict(params); merged.update(kw)  # allow overrides
            super().__init__(index, **merged)
    _Alias.__name__ = name  # registry uses class.__name__ as key
    return name

def register_from_json(base_name: str, json_path: str, alias: Optional[str] = None) -> str:
    with open(json_path, "r") as f:
        data = json.load(f)
    params = _extract_genome(data)
    if not params:
        raise ValueError(f"No numeric params found in {json_path}")
    # If file has a 'name', prefer that as alias
    tag = alias or data.get("name")
    return register_alias(base_name, params, tag)

def autoload(pattern: str = "zoo/**/*.json", base_name: Optional[str] = None) -> Dict[str, str]:
    out = {}
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            candidate = data.get("candidate") or data.get("class")
            if base_name is not None and candidate != base_name:
                continue
            params = _extract_genome(data)
            if not params:
                continue
            alias = os.path.splitext(os.path.basename(path))[0]
            reg_name = register_alias(candidate or base_name, params, alias=alias)
            out[path] = reg_name
        except Exception:
            continue
    return out
