#!/usr/bin/env python3
"""Prove the neighbor-on and neighbor-off envs share one identical physical reset.

For each checked seed, a neighbor-ON env (18-dim obs) and a neighbor-OFF env
(11-dim obs) are reset with the same seed and every physical state field and the
env RNG stream are compared exactly. Rendering is then shown to advance neither
the RNG nor the physics: the full post-reset state is snapshotted, ``render()`` is
called, and the state is re-checked bit-for-bit.

Writes ``results/identity_evidence.json``. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

import demo_common as C  # noqa: E402

FIELDS = [
    "fish_positions", "fish_velocities", "fish_alive",
    "fish_death_timesteps", "predator_pos", "predator_vel", "timestep",
]


def snapshot(env):
    snap = {f: getattr(env, f) for f in FIELDS}
    snap["np_random_state"] = env.np_random.bit_generator.state
    return snap


def equal(a, b):
    for f in FIELDS:
        if not np.array_equal(a[f], b[f]):
            return False, f
    if a["np_random_state"] != b["np_random_state"]:
        return False, "np_random_state"
    return True, None


def check_seed(seed, results):
    e_on = C.make_env(True)
    e_off = C.make_env(False)
    rec = {"seed": int(seed)}
    try:
        o_on, _ = e_on.reset(seed=int(seed))
        o_off, _ = e_off.reset(seed=int(seed))
        a, b = snapshot(e_on), snapshot(e_off)
        ok_state, bad = equal(a, b)
        rec["reset_state_identical"] = ok_state
        rec["reset_state_first_mismatch"] = bad
        rec["base_obs_identical"] = bool(np.array_equal(np.asarray(o_on)[:, :11], np.asarray(o_off)[:, :11]))
        rec["obs_shapes"] = [list(np.asarray(o_on).shape), list(np.asarray(o_off).shape)]

        # Rendering must not advance the RNG or the physics.
        e_on.render_mode = "rgb_array"
        before = snapshot(e_on)
        frame = e_on.render()
        after = snapshot(e_on)
        ok_render, bad_render = equal(before, after)
        rec["render_state_unchanged"] = ok_render
        rec["render_state_first_mismatch"] = bad_render
        rec["render_frame_shape"] = list(np.asarray(frame).shape)
    finally:
        e_on.close()
        e_off.close()
    results.append(rec)
    return rec


def main():
    seeds = [C.report_seeds()[0]] + C.debug_seeds() + [1, 123456789]
    results = []
    for s in seeds:
        check_seed(s, results)
    ok = all(
        r["reset_state_identical"] and r["base_obs_identical"] and r["render_state_unchanged"]
        for r in results
    )
    evidence = {
        "kind": "before_after_identity_evidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "neighbor-on (18-dim) and neighbor-off (11-dim) envs share one identical "
                 "physical reset and RNG stream; rendering changes neither",
        "fields_compared": FIELDS + ["np_random_state"],
        "checked_seeds": seeds,
        "records": results,
        "all_identical": ok,
    }
    out = C.DEMO_DIR / "results" / "identity_evidence.json"
    out.write_text(json.dumps(evidence, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "obs_shapes"} for r in results], indent=2))
    print(f"[written] {out}")
    if not ok:
        raise SystemExit("IDENTITY CHECK FAILED")
    print("IDENTITY CHECK PASSED")


if __name__ == "__main__":
    main()
