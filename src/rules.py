# =============================================================================
# src/rules.py — Rule engine for Paper A (RGL); reused in Paper B (IBD)
#
# Grades a firm's ratios (3 directions), flags deteriorating trends for
# Profitability/Solvency, computes a composite risk score (category x signal
# tier), and extracts top driver "cause accounts". Config:
# configs/rule_thresholds_A.yaml.
#
# Data: Drotar et al. (2019), DiB 25:104360; Mendeley V2 (CC BY 4.0).
# Ratio identities: Zoricak et al. (2020), Econ. Modelling 84:165-176, Table 2.
# Textbook cutoffs: Li (2024), Accounting & Finance.
# =============================================================================
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

ABBR2ID = {
    "ROA": 1, "ROE": 2, "ROS": 3, "L1": 4, "L2": 5, "L3": 6, "TAT": 7,
    "ATD": 8, "DTR": 9, "ITD": 10, "DA": 11, "DE": 12, "FL": 13, "ROI": 14,
    "DIR": 15, "DCR": 16, "ACR": 17, "LP": 18, "BL": 19, "LRR": 20, "WAR": 21,
}
GRADE_SEVERITY = {"normal": 0.0, "watch": 0.5, "alert": 1.0, "na": 0.0}


def _find_root() -> Path:
    p = Path.cwd()
    if p.name in {"notebooks", "paperA", "paperB", "paperC", "paperD"}:
        return p.parents[0] if p.name == "notebooks" else p.parents[1]
    return p


@dataclass
class RuleConfig:
    thresholds: dict
    category_weights: dict

    @classmethod
    def load(cls, path: Path | None = None) -> "RuleConfig":
        root = _find_root()
        fp = path or (root / "configs" / "rule_thresholds_A.yaml")
        try:
            import yaml
        except ModuleNotFoundError as e:
            raise RuntimeError("pyyaml required. `pip install pyyaml`.") from e
        y = yaml.safe_load(fp.read_text(encoding="utf-8"))
        th = {}
        for ab, d in y["thresholds"].items():
            watch = d["watch"]
            th[ab] = {
                "direction": d["direction"],
                "watch": None if watch in (None, "null") else float(watch),
                "alert": float(d["alert"]),
                "category": d["category"],
                "signal_tier": int(d["signal_tier"]),
            }
        return cls(thresholds=th, category_weights=dict(y["category_weights"]))


def grade_value(value: float, direction: str, watch, alert) -> str:
    if pd.isna(value):
        return "na"
    if direction == "low_bad":
        if value <= alert: return "alert"
        if watch is not None and value <= watch: return "watch"
        return "normal"
    if direction == "high_bad":
        if value >= alert: return "alert"
        if watch is not None and value >= watch: return "watch"
        return "normal"
    if direction == "negative_bad":
        if value <= alert: return "alert"
        if watch is not None and value <= watch: return "watch"
        return "normal"
    raise ValueError(f"unknown direction: {direction}")


def trend_direction(y1, y2, y3, worsen_if: str) -> str:
    ys = np.array([y1, y2, y3], float)
    ok = ~np.isnan(ys)
    if ok.sum() < 2:
        return "unknown"
    slope = np.polyfit(np.array([0., 1., 2.])[ok], ys[ok], 1)[0]
    if abs(slope) < 1e-9:
        return "flat"
    if worsen_if == "down":
        return "worsening" if slope < 0 else "improving"
    return "worsening" if slope > 0 else "improving"


@dataclass
class FirmAssessment:
    grades: dict = field(default_factory=dict)
    trends: dict = field(default_factory=dict)
    contributions: dict = field(default_factory=dict)
    composite_risk: float = 0.0
    drivers: list = field(default_factory=list)


def assess_firm(row: pd.Series, cfg: RuleConfig, top_k: int = 5) -> FirmAssessment:
    grades, trends, contribs = {}, {}, {}
    for ab, spec in cfg.thresholds.items():
        rid = ABBR2ID[ab]
        v3 = row.get(f"ratio{rid:02d}_y3", np.nan)
        g = grade_value(v3, spec["direction"], spec["watch"], spec["alert"])
        grades[ab] = g
        if spec["category"] in ("Profitability", "Solvency"):
            worsen = "down" if spec["direction"] in ("low_bad", "negative_bad") else "up"
            trends[ab] = trend_direction(
                row.get(f"ratio{rid:02d}_y1", np.nan),
                row.get(f"ratio{rid:02d}_y2", np.nan), v3, worsen)
        w = cfg.category_weights.get(spec["category"], 0.5) * spec["signal_tier"]
        contribs[ab] = GRADE_SEVERITY[g] * w
    total_w = sum(cfg.category_weights.get(s["category"], 0.5) * s["signal_tier"]
                  for s in cfg.thresholds.values())
    composite = sum(contribs.values()) / total_w if total_w else 0.0
    drivers = [ab for ab, _ in sorted(contribs.items(),
               key=lambda kv: kv[1], reverse=True) if contribs[ab] > 0][:top_k]
    return FirmAssessment(grades=grades, trends=trends, contributions=contribs,
                          composite_risk=composite, drivers=drivers)


def assess_frame(df: pd.DataFrame, cfg: RuleConfig, top_k: int = 5) -> pd.DataFrame:
    out = []
    for _, row in df.iterrows():
        a = assess_firm(row, cfg, top_k)
        out.append({"composite_risk": a.composite_risk,
                    "n_alert": sum(g == "alert" for g in a.grades.values()),
                    "n_watch": sum(g == "watch" for g in a.grades.values()),
                    "drivers": "|".join(a.drivers)})
    res = pd.DataFrame(out, index=df.index)
    for c in ("bankrupt", "industry", "eval_year"):
        if c in df.columns:
            res[c] = df[c].values
    return res
