#!/usr/bin/env python3
"""Freeze the demo protocol BEFORE any score is produced.

Writes ``protocol_binding.json`` (machine-readable) and ``protocol.md``
(human-readable) recording: the full unchanged eval env config, every
controller's checkpoint path + loaded policy-tensor SHA-256, the per-controller
observation-length difference and its neighbor flag, and both frozen seed banks.
The NEW model hashes are additionally checked against the ``final_policy_tensor_sha256``
fields of the accepted v50 corrected selection manifest.

Refuses to overwrite an existing protocol unless --allow-overwrite is given.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import demo_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    out_json = C.DEMO_DIR / "results" / "protocol_binding.json"
    out_md = C.DEMO_DIR / "protocol.md"
    if (out_json.exists() or out_md.exists()) and not args.allow_overwrite:
        raise SystemExit("protocol already exists; refusing to overwrite (use --allow-overwrite)")

    report = C.report_seeds()
    debug = C.debug_seeds()

    # Disjointness sanity: our fresh banks vs every prior bank, and mutually.
    prior_values = set()
    for rng_seed, n in C.PRIOR_BANKS:
        prior_values.update(C.episode_seeds(rng_seed, n))
    overlap_report = sorted(set(report) & prior_values)
    overlap_debug = sorted(set(debug) & prior_values)
    mutual = sorted(set(report) & set(debug))
    assert not overlap_report, f"report bank overlaps a prior bank: {overlap_report[:5]}"
    assert not overlap_debug, f"debug bank overlaps a prior bank: {overlap_debug[:5]}"
    assert not mutual, f"report and debug banks are not disjoint: {mutual[:5]}"

    # Model identities.
    manifest = json.loads((C.ROOT / C.CORRECTED_SELECTION_MANIFEST_RELPATH).read_text())
    new_hash_check = {}
    for r in (0, 1, 2):
        rel = C.NEW_MODEL_RELPATH[r]
        h = C.policy_tensor_sha256_from_zip(C.ROOT / rel)
        frozen = manifest["selected"][f"rep{r}_survival_only"]["final_policy_tensor_sha256"]
        ok = (h == frozen)
        new_hash_check[f"new_rep{r}"] = {"computed": h, "manifest_final": frozen, "match": ok}
        assert ok, f"new_rep{r} tensor hash {h} != manifest final {frozen}"
    old_hash = C.policy_tensor_sha256_from_zip(C.ROOT / C.OLD_MODEL_RELPATH)

    binding = {
        "kind": "before_after_demo_protocol_binding",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "freeze recorded before any benchmark score; see protocol.md",
        "physics": "unchanged FishEscapeEnv via experiments/v50/v50_common.env_config",
        "env_config": C.v50.env_config(include_neighbor_features=False),
        "num_fish": C.NUM_FISH,
        "max_timesteps": C.MAX_TIMESTEPS,
        "dt": 0.1,
        "phase_steps": list(C.PHASE_STEPS),
        "condition": "nominal same-scene initial state (initial_escape_boost=True, escape_boost_speed=0.8)",
        "controllers": [
            {
                "name": c["name"],
                "kind": c["kind"],
                "rule": c.get("rule"),
                "include_neighbor_features": c["neighbor"],
                "obs_dim": 18 if c["neighbor"] else 11,
                "checkpoint": c.get("path"),
                "policy_tensor_sha256": (
                    old_hash if c["name"] == "old_ppo"
                    else new_hash_check[c["name"]]["computed"] if c["name"] in new_hash_check
                    else None
                ),
                "deterministic": True,
            }
            for c in C.CONTROLLERS
        ],
        "new_hash_matches_manifest_final": new_hash_check,
        "report_bank": C.REPORT_BANK,
        "debug_bank": C.DEBUG_BANK,
        "report_seeds": report,
        "debug_seeds": debug,
        "disjointness": {
            "report_vs_prior": overlap_report,
            "debug_vs_prior": overlap_debug,
            "report_vs_debug": mutual,
        },
        "statistic": "mean over scenarios of final_alive/96; fixed /96 denominator; no episode dropped",
        "primary_contrast": "new_fixed3_mean(scene) - old(scene), scene-bootstrap 10000",
    }
    out_json.write_text(json.dumps(binding, indent=2))

    report_hash = C.sha256_file(out_json)

    def ctrl_table():
        rows = []
        for c in binding["controllers"]:
            h = c["policy_tensor_sha256"]
            h = (h[:16] + "...") if h else "(rule)"
            rows.append(
                f"| {c['name']} | {c['kind']} | {c['obs_dim']} | "
                f"{'on' if c['include_neighbor_features'] else 'off'} | "
                f"{c['checkpoint'] or '(rule logic)'} | {h} |"
            )
        return "\n".join(rows)

    md = f"""# Before/after same-scene benchmark — frozen protocol

Created (UTC): {binding['created_utc']}
Protocol binding SHA-256: `{report_hash}`

This protocol was written and hashed **before any benchmark score was computed**.
Physics, reward, termination and reset are the unchanged v50 world
(`experiments/v50/v50_common.env_config`); this is not a re-training and no
checkpoint is re-selected on any demo result.

## Question

On one fresh matched scenario bank, compared against the historical best single
checkpoint, how does the current three-model method perform end to end, and where
does the accepted hand-written evasion rule sit? This is a whole-method
comparison on identical physical resets. It is not an isolated training-bug fix,
not a pure-reward ablation, and the old side is a single representative checkpoint
(not three runs).

## Controllers (all deterministic)

| name | kind | obs dim | neighbor | checkpoint | policy-tensor SHA-256 |
|---|---|---|---|---|---|
{ctrl_table()}

`old_ppo` uses `include_neighbor_features=True` only so its observation matches
its own 18-dim input; the three NEW models and the rule run at 11 dims. This flag
changes the observation vector length and nothing else in the simulation.

NEW model hashes were checked against the `final_policy_tensor_sha256` fields of
`{C.CORRECTED_SELECTION_MANIFEST_RELPATH}` (all match; recorded in the binding).

## Scenes

- report bank: `default_rng({C.REPORT_BANK['rng_seed']})`, {C.REPORT_BANK['n']} scenarios
- debug bank: `default_rng({C.DEBUG_BANK['rng_seed']})`, {C.DEBUG_BANK['n']} scenarios

Each scene is one shared world: the same seed drives every controller, and the
simulation RNG / positions / velocities / collision physics are identical across
the neighbor-on and neighbor-off envs (asserted in `initial_state_identity.py`
before scoring).

## Runs

`{len(C.CONTROLLERS)} controllers x {C.REPORT_BANK['n']} scenarios = {len(C.CONTROLLERS) * C.REPORT_BANK['n']} episodes`
on the report bank, plus a 2-scene debug bank. Survival is recorded as
`num_alive / 96` at steps {list(C.PHASE_STEPS)}; the denominator is always 96, no
episode is dropped, and a first-step death is kept.

## Statistics

- per-controller mean final survival over the 64 scenarios;
- `new_fixed3_mean` = per-scene mean of the three NEW replicates;
- paired contrast: per-scene `new_fixed3_mean - old`, then scenario bootstrap
  (10000) 95% CI; the same for `rule - old` and `rule - new_fixed3_mean`.
"""
    out_md.write_text(md)
    print(f"[written] {out_json}\n[written] {out_md}\n[protocol sha256] {report_hash}")


if __name__ == "__main__":
    main()
