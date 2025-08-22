
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import math

CATALOG: Dict[str, Dict[str, Dict[str, float|str]]] = {
    "AegisPilot": {
        "partner_repel_w": {"lo": 0.0, "hi": 2.0, "type": "float"},
        "pair_repel_w":    {"lo": 0.0, "hi": 4.0, "type": "float"},
        "red_attr_w":      {"lo": 0.2, "hi": 3.0, "type": "float"},
        "green_attr_w":    {"lo": 0.0, "hi": 1.5, "type": "float"},
        "wall_power":      {"lo": 2.0, "hi": 8.0, "type": "float"},
        "wall_scale":      {"lo": 0.05,"hi": 0.25,"type": "float"},
        "sample_ahead":    {"lo": 0.01,"hi": 0.15,"type": "float"},
        "fd_step":         {"lo": 0.002,"hi": 0.02,"type": "float"},
    },
    "ConeKeeper": {
        "samples_heading": {"lo": 5, "hi": 15, "type": "int"},
        "engage_radius":   {"lo": 0.10, "hi": 0.20, "type": "float"},
        "vo_time_horizon": {"lo": 0.4, "hi": 1.0, "type": "float"},
        "vo_weight":       {"lo": 0.6, "hi": 2.2, "type": "float"},
        "safety_radius":   {"lo": 0.08, "hi": 0.16, "type": "float"},
        "wall_weight":     {"lo": 0.6, "hi": 1.6, "type": "float"},
        "horizon":         {"lo": 0.35, "hi": 0.8, "type": "float"},
        "dt":              {"lo": 0.06, "hi": 0.12, "type": "float"},
        "v_max":           {"lo": 0.8, "hi": 1.2, "type": "float"},
        "w_max":           {"lo": 2.0, "hi": 4.0, "type": "float"},
    },
    "AuctionStrider": {
        "early_bonus":     {"lo": 0.2, "hi": 1.0, "type": "float"},
        "early_decay":     {"lo": 4.0, "hi": 16.0, "type": "float"},
        "commit_time":     {"lo": 0.5, "hi": 1.2, "type": "float"},
        "switch_threshold":{"lo": 0.08,"hi": 0.25,"type": "float"},
        "samples_v":       {"lo": 3,   "hi": 6,   "type": "int"},
        "samples_w":       {"lo": 5,   "hi": 11,  "type": "int"},
        "horizon":         {"lo": 0.3, "hi": 0.7, "type": "float"},
        "clearance_radius":{"lo": 0.08,"hi": 0.16,"type": "float"},
        "corner_penalty_w":{"lo": 0.2, "hi": 1.2, "type": "float"},
        "v_max":           {"lo": 0.8, "hi": 1.2, "type": "float"},
        "w_max":           {"lo": 2.0, "hi": 4.0, "type": "float"},
    },
    "TerritoryDash": {
        "zone_margin":     {"lo": 0.0, "hi": 0.15, "type": "float"},
        "near_band":       {"lo": 0.05,"hi": 0.30, "type": "float"},
        "hook_offset":     {"lo": 0.10,"hi": 0.22, "type": "float"},
        "hook_swing_time": {"lo": 0.30,"hi": 0.70, "type": "float"},
        "run_speed":       {"lo": 0.7, "hi": 1.2,  "type": "float"},
        "w_limit":         {"lo": 2.0, "hi": 4.0,  "type": "float"},
    },
    "CutoffCaptain": {
        "intercept_horizon":{"lo":0.6,"hi":1.6,"type":"float"},
        "stall_eps":       {"lo":0.02,"hi":0.10,"type":"float"},
        "stall_time":      {"lo":0.5, "hi":1.2,"type":"float"},
        "sweep_speed":     {"lo":0.4, "hi":1.0,"type":"float"},
        "sweep_turn":      {"lo":1.4, "hi":3.0,"type":"float"},
        "commit_time":     {"lo":0.5, "hi":1.2,"type":"float"},
    },
    "FieldMarshal": {
        "partner_repel_w": {"lo": 0.0, "hi": 0.6, "type": "float"},
        "pair_repel_w":    {"lo": 0.8, "hi": 3.0, "type": "float"},
        "red_attr_w":      {"lo": 0.6, "hi": 1.8, "type": "float"},
        "green_attr_w":    {"lo": 0.0, "hi": 0.8, "type": "float"},
        "wall_power":      {"lo": 3.0, "hi": 7.0, "type": "float"},
        "wall_scale":      {"lo": 0.05,"hi": 0.20,"type": "float"},
        "sample_ahead":    {"lo": 0.02,"hi": 0.12,"type": "float"},
        "fd_step":         {"lo": 0.003,"hi":0.02,"type": "float"},
        "commit_time":     {"lo": 0.4, "hi": 1.2, "type": "float"},
        "switch_threshold":{"lo": 0.08,"hi": 0.25,"type": "float"},
    },
    "FuzzyPusher": {
        "w_max":           {"lo": 2.0, "hi": 4.0, "type": "float"},
        "v_max":           {"lo": 0.7, "hi": 1.2, "type": "float"},
        "commit_time":     {"lo": 0.4, "hi": 1.2, "type": "float"},
        "switch_threshold":{"lo": 0.08,"hi": 0.25,"type": "float"},
    },
}

def _infer_from_defaults(defaults):
    spec = {}
    for k, v in defaults.items():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (list, tuple, dict)):
            continue
        if isinstance(v, (int, float)):
            name = k.lower()
            lo, hi = None, None
            typ = "int" if isinstance(v, int) else "float"
            if "angle" in name and "deg" in name: lo, hi = 10.0, 80.0
            elif "angle" in name: lo, hi = 0.2, 3.2
            elif name.endswith("_w") or "weight" in name: lo, hi = 0.0, max(1.0, (v*3.0 if v>0 else 3.0))
            elif "power" in name: lo, hi = 2.0, 8.0
            elif "scale" in name: lo, hi = 0.02, 0.3
            elif "radius" in name: lo, hi = 0.05, 0.6
            elif "time" in name or "horizon" in name or "timeout" in name: lo, hi = 0.2, 1.5
            elif name in ("v_max","run_speed","hook_speed","unstick_speed"): lo, hi = 0.5, 1.2
            elif name in ("w_max","w_limit","turn","sweep_turn"): lo, hi = 1.5, 4.0
            elif name in ("dt","fd_step","sample_ahead"): lo, hi = 0.002, 0.2
            elif "samples" in name or "count" in name: typ = "int"; lo, hi = 3, 15
            else:
                span = 2.0*abs(float(v)) if v != 0 else 1.0
                lo, hi = float(v) - 0.5*span, float(v) + 0.5*span
            spec[k] = {"lo": float(lo), "hi": float(hi), "type": typ}
    return spec

def normalize_spec(spec_in):
    out = {}
    for k, v in spec_in.items():
        if isinstance(v, dict):
            lo, hi = float(v["lo"]), float(v["hi"]); typ = v.get("type","float")
        elif isinstance(v, (list, tuple)) and len(v)==2:
            lo, hi = float(v[0]), float(v[1]); typ = "float"
        else:
            raise ValueError(f"Bad spec entry for {k}: {v}")
        out[k] = {"lo": lo, "hi": hi, "type": typ}
    return out

def get_param_spec_for_class(cls):
    attr = getattr(cls, "PARAM_SPEC", None)
    if attr:
        return normalize_spec(attr)
    name = getattr(cls, "__name__", str(cls))
    if name in CATALOG:
        return normalize_spec(CATALOG[name])
    defaults = getattr(cls, "DEFAULTS", {}) or {}
    return normalize_spec(_infer_from_defaults(defaults))

def clip_params(spec, params):
    out = {}
    for k, meta in spec.items():
        lo, hi = meta["lo"], meta["hi"]
        v = params.get(k, (lo+hi)/2.0)
        v = max(lo, min(hi, float(v)))
        if meta.get("type") == "int":
            v = int(round(v))
        out[k] = v
    return out

def params_to_vec01(spec, params):
    x = []
    for k, meta in spec.items():
        lo, hi = meta["lo"], meta["hi"]
        v = params.get(k, (lo+hi)/2.0)
        u = (float(v) - lo) / (hi - lo + 1e-12)
        x.append(max(0.0, min(1.0, u)))
    return x

def vec01_to_params(spec, x01):
    keys = list(spec.keys())
    out = {}
    for i, k in enumerate(keys):
        meta = spec[k]
        lo, hi = meta["lo"], meta["hi"]
        u = max(0.0, min(1.0, float(x01[i])))
        v = lo + u * (hi - lo)
        if meta.get("type") == "int":
            v = int(round(v))
        out[k] = v
    return out

def spec_keys(spec):
    return list(spec.keys())
