#!/usr/bin/env python3
"""Independent verification for the before/after demo.

Checks, without importing the analysis summariser's internals:
  * the two frozen banks are disjoint from every prior bank and from each other;
  * every protocol-bound model tensor hash still matches (NEW vs the accepted v50
    corrected selection manifest final fields);
  * the report summary numbers recompute exactly from the raw jsonl;
  * the video's first-scenario seed replays with a shared initial state and a
    shared predator trajectory across the three panels, and reports each panel's
    final alive/96.

Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

import demo_common as C  # noqa: E402

checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}")


def main():
    # 1. banks
    prior = set()
    for rs, n in C.PRIOR_BANKS:
        prior.update(C.episode_seeds(rs, n))
    rep, dbg = C.report_seeds(), C.debug_seeds()
    check("report bank disjoint from prior", not (set(rep) & prior))
    check("debug bank disjoint from prior", not (set(dbg) & prior))
    check("report/debug mutually disjoint", not (set(rep) & set(dbg)))

    # 2. model hashes
    man = json.loads((C.ROOT / C.CORRECTED_SELECTION_MANIFEST_RELPATH).read_text())
    binding = json.loads((C.DEMO_DIR / "results" / "protocol_binding.json").read_text())
    by_name = {c["name"]: c for c in binding["controllers"]}
    for r in (0, 1, 2):
        h = C.policy_tensor_sha256_from_zip(C.ROOT / C.NEW_MODEL_RELPATH[r])
        frozen = man["selected"][f"rep{r}_survival_only"]["final_policy_tensor_sha256"]
        check(f"new_rep{r} hash == manifest final", h == frozen == by_name[f"new_rep{r}"]["policy_tensor_sha256"])

    # 3. independent recompute from raw
    by = {}
    for line in (C.DEMO_DIR / "results" / "report_raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        by.setdefault(r["controller"], {})[int(r["seed"])] = r["final_survival"]
    seeds = C.report_seeds()
    surv = {n: np.array([by[n][s] for s in seeds]) for n in C.CONTROLLER_NAMES}
    new_mean = float(np.mean([surv[n] for n in C.NEW_REPS]))
    summary = json.loads((C.DEMO_DIR / "results" / "report_summary.json").read_text())
    check("new_fixed3_mean recompute", abs(new_mean - summary["new_fixed3_mean"]["mean_final_survival"]) < 1e-12,
          f"{new_mean:.6f}")
    for n in C.CONTROLLER_NAMES:
        check(f"{n} mean recompute", abs(surv[n].mean() - summary["per_controller"][n]["mean_final_survival"]) < 1e-12)

    # 4. replay the video's first scenario, assert shared world/predator
    seed = rep[0]
    from stable_baselines3 import PPO

    specs = [
        {"name": "old_ppo", "neighbor": True, "model": PPO.load(str(C.ROOT / C.OLD_MODEL_RELPATH), device="cpu")},
        {"name": "new_rep0", "neighbor": False, "model": PPO.load(str(C.ROOT / C.NEW_MODEL_RELPATH[0]), device="cpu")},
        {"name": "rule_flee_lead", "neighbor": False, "model": None},
    ]
    envs = [C.make_env(s["neighbor"]) for s in specs]
    obs = []
    for e in envs:
        o, _ = e.reset(seed=int(seed))
        obs.append(np.asarray(o))
    shared0 = all(
        np.array_equal(envs[0].fish_positions, e.fish_positions)
        and np.array_equal(envs[0].predator_pos, e.predator_pos)
        for e in envs[1:]
    )
    shared_pred = True
    for _ in range(500):
        for j, s in enumerate(specs):
            if len(obs[j]) > 0:
                if s["model"] is not None:
                    b, _ = s["model"].predict(obs[j], deterministic=True)
                    acts = [int(a) for a in np.atleast_1d(b)]
                else:
                    acts = [C.rule_action("flee_lead", obs[j][i]) for i in range(len(obs[j]))]
            else:
                acts = []
            obs[j], _r, term, trunc, info = envs[j].step(acts)
            obs[j] = np.asarray(obs[j])
        shared_pred = shared_pred and all(np.array_equal(envs[0].predator_pos, e.predator_pos) for e in envs[1:])
    for e in envs:
        e.close()
    check("video seed shared initial state", shared0, f"seed={seed}")
    check("video seed shared predator trajectory", shared_pred)

    alive = {}
    # final alive from the last step of each env is lost after close; recompute quickly
    envs2 = [C.make_env(s["neighbor"]) for s in specs]
    obs2 = []
    for e in envs2:
        o, _ = e.reset(seed=int(seed))
        obs2.append(np.asarray(o))
    final = None
    for _ in range(500):
        for j, s in enumerate(specs):
            if len(obs2[j]) > 0:
                if s["model"] is not None:
                    b, _ = s["model"].predict(obs2[j], deterministic=True)
                    acts = [int(a) for a in np.atleast_1d(b)]
                else:
                    acts = [C.rule_action("flee_lead", obs2[j][i]) for i in range(len(obs2[j]))]
            else:
                acts = []
            obs2[j], _r, term, trunc, info = envs2[j].step(acts)
            obs2[j] = np.asarray(obs2[j])
            if j == 0:
                final = info
    for s, e in zip(specs, envs2):
        alive[s["name"]] = int(np.sum(e.fish_alive))
    for e in envs2:
        e.close()
    check("video sample final alive recorded", True, json.dumps(alive))

    out = {
        "kind": "before_after_demo_verification",
        "checks": checks,
        "all_pass": all(c["ok"] for c in checks),
        "video_seed": int(seed),
        "video_sample_final_alive": alive,
    }
    (C.DEMO_DIR / "results" / "verification.json").write_text(json.dumps(out, indent=2))
    print(f"[written] results/verification.json  all_pass={out['all_pass']}")
    if not out["all_pass"]:
        raise SystemExit("VERIFICATION FAILED")


if __name__ == "__main__":
    main()
