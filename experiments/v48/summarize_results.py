#!/usr/bin/env python3
"""Consolidate v48 stage summaries into one comparison table (no recomputation)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "artifacts" / "results"

SUMMARY_FILES = [
    "smoke.summary.json", "dev.summary.json", "report.summary.json",
    "report_extra.summary.json", "tau_sweep_dev.summary.json",
    "legacy_report.summary.json", "phase_dev.summary.json",
    "safe_top_dev.summary.json", "phase_report.summary.json",
    "phase_report2.summary.json",
]

out = {"stages": {}, "reward_vs_survival": {}, "predator_probe": None,
       "phase_analysis": None}
for fn in SUMMARY_FILES:
    p = RES / fn
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    out["stages"][fn.replace(".summary.json", "")] = {
        "n_seeds": d["n_seeds"],
        "seed_set": d.get("seed_set"),
        "wall_sec": d["wall_sec"],
        "arms": {k: {kk: v[kk] for kk in
                     ("mean_final_survival", "std", "min", "max",
                      "mean_survival_at_step", "mean_action_dist",
                      "predator_second_half")
                     if kk in v}
                 for k, v in d["arms"].items()},
        "paired_vs_reference": d.get("paired_vs_reference", {}),
    }

rows = [json.loads(l) for l in (RES / "report.jsonl").read_text().splitlines()]
agg = {}
for r in rows:
    a = agg.setdefault(r["control"], {"surv": [], "rew": []})
    a["surv"].append(r["final_survival"])
    a["rew"].append(r["avg_reward"])
for k, a in agg.items():
    out["reward_vs_survival"][k] = {
        "mean_final_survival": sum(a["surv"]) / len(a["surv"]),
        "mean_avg_reward": sum(a["rew"]) / len(a["rew"]),
    }

for name in ("predator_probe.json", "phase_analysis.json", "probe_policy_space.json"):
    p = RES / name
    if p.exists():
        out[name.replace(".json", "")] = json.loads(p.read_text())

(RES / "consolidated.json").write_text(json.dumps(out, indent=2))
print("[written]", RES / "consolidated.json")
print("stages:", list(out["stages"]))
