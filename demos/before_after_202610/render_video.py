#!/usr/bin/env python3
"""Render the three-panel before/after gameplay video for one first-bank scene.

Scene: the FIRST scenario of the frozen 64-scenario report bank (chosen before any
score was seen -- it is bank index 0, not the scene the new model wins most).

Three panels advance in lockstep on one identical physical reset (asserted):
  old PPO (18-dim, neighbor on) | new PPO rep0 (11-dim) | rule_flee_lead.
The predator is action-independent, so its trajectory is asserted identical across
the three panels every step. Nothing is faked: every drawn position is the true
env state after the real step.

Render layer only: a faithful projection (circle boundary, radii proportional to
FISH_SIZE / PREDATOR_SIZE, field margin). Physics and the env are the unchanged
FishEscapeEnv. 500 steps at dt=0.1 -> 500 frames at 10 fps = 50 s; frame i is
step i.

Writes a fresh run directory (refuses to overwrite a non-empty one) with the mp4,
five posters, a contact sheet, and the frame sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import demo_common as C  # noqa: E402

FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
W, H = 1800, 800
PANEL_W = 600
CIRCLE_CY = 434
CIRCLE_R = 244
TITLE_Y, SUBTITLE_Y, LEGEND_Y, LABEL_Y = 10, 52, 84, 116
ALIVE_Y, TIMER_Y, FOOTER_Y = 680, 710, 758
C_BG = (255, 255, 255)
C_BOUND = (185, 185, 185)
C_FISH = (0, 150, 255)
C_PRED = (255, 0, 0)
C_TEXT = (25, 25, 25)
C_SUB = (85, 85, 85)
STAGE_RADIUS = 10.0
FISH_SIZE = 0.2
PREDATOR_SIZE = 0.5
PX_PER_UNIT = CIRCLE_R / STAGE_RADIUS
FISH_R = max(2, int(round(FISH_SIZE * PX_PER_UNIT)))
PRED_R = max(3, int(round(PREDATOR_SIZE * PX_PER_UNIT)))
POSTER_TIMES = [0.1, 10.0, 25.0, 40.0, 50.0]


def load_copy():
    return json.loads((C.DEMO_DIR / "publiccopy.json").read_text())


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def ctext(draw, xy, text, fnt, fill, anchor="mm"):
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def world_to_screen(x, y, panel_idx):
    cx = panel_idx * PANEL_W + PANEL_W // 2
    sx = cx + (x / STAGE_RADIUS) * CIRCLE_R
    sy = CIRCLE_CY - (y / STAGE_RADIUS) * CIRCLE_R
    return sx, sy


def build_base(copy):
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    f_title, f_sub, f_label, f_body = font(36), font(24), font(27), font(24)
    ctext(d, (W // 2, TITLE_Y + 20), copy["title"], f_title, C_TEXT)
    ctext(d, (W // 2, SUBTITLE_Y + 14), copy["subtitle"], f_sub, C_SUB)
    ctext(d, (W // 2, LEGEND_Y + 14), f"{copy['legend_fish']}    {copy['legend_predator']}", f_sub, C_SUB)
    labels = [copy["old_label"], copy["new_label"], copy["rule_label"]]
    for i, lab in enumerate(labels):
        cx = i * PANEL_W + PANEL_W // 2
        ctext(d, (cx, LABEL_Y + 15), lab, f_label, C_TEXT)
        d.ellipse([cx - CIRCLE_R, CIRCLE_CY - CIRCLE_R, cx + CIRCLE_R, CIRCLE_CY + CIRCLE_R],
                  outline=C_BOUND, width=2)
    ctext(d, (W // 2, FOOTER_Y + 15), f"{copy['sample_note']}  ·  {copy['benchmark_note']}", f_body, C_SUB)
    return img, d, f_body


def draw_panel(img, d, panel_idx, env, copy, f_body):
    alive = env.fish_alive
    for i in range(env.NUM_FISH):
        if alive[i]:
            sx, sy = world_to_screen(float(env.fish_positions[i][0]), float(env.fish_positions[i][1]), panel_idx)
            d.ellipse([sx - FISH_R, sy - FISH_R, sx + FISH_R, sy + FISH_R], fill=C_FISH)
    px, py = world_to_screen(float(env.predator_pos[0]), float(env.predator_pos[1]), panel_idx)
    d.ellipse([px - PRED_R, py - PRED_R, px + PRED_R, py + PRED_R], fill=C_PRED)
    n = int(alive.sum())
    cx = panel_idx * PANEL_W + PANEL_W // 2
    ctext(d, (cx, ALIVE_Y + 14), f"{copy['survivors_label']} {n}/{env.NUM_FISH} ({100.0 * n / env.NUM_FISH:.0f}%)", f_body, C_TEXT)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", type=int, default=0, help="render only N frames (layout check)")
    args = ap.parse_args()

    copy = load_copy()
    seed = C.report_seeds()[0]
    stamp = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = C.DEMO_DIR / "video" / stamp
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty run dir: {run_dir}")
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "posters").mkdir(exist_ok=True)

    from stable_baselines3 import PPO

    specs = [
        {"name": "old_ppo", "neighbor": True, "model": PPO.load(str(C.ROOT / C.OLD_MODEL_RELPATH), device="cpu")},
        {"name": "new_rep0", "neighbor": False, "model": PPO.load(str(C.ROOT / C.NEW_MODEL_RELPATH[0]), device="cpu")},
        {"name": "rule_flee_lead", "neighbor": False, "model": None},
    ]
    envs = [C.make_env(s["neighbor"]) for s in specs]
    for e in envs:
        e.render_mode = "rgb_array"

    obs = []
    for e in envs:
        o, _ = e.reset(seed=int(seed))
        obs.append(np.asarray(o))

    # Assert one identical physical reset across the three panels.
    ref = envs[0]
    for j, e in enumerate(envs[1:], 1):
        assert np.array_equal(ref.fish_positions, e.fish_positions), f"panel{j} fish pos differs at reset"
        assert np.array_equal(ref.fish_velocities, e.fish_velocities), f"panel{j} fish vel differs at reset"
        assert np.array_equal(ref.fish_alive, e.fish_alive), f"panel{j} fish alive differs at reset"
        assert np.array_equal(ref.predator_pos, e.predator_pos), f"panel{j} predator pos differs at reset"
        assert np.array_equal(ref.predator_vel, e.predator_vel), f"panel{j} predator vel differs at reset"

    base, bd, f_body = build_base(copy)
    steps = args.smoke if args.smoke else args.steps
    posters_at = {int(round(t * args.fps)): t for t in POSTER_TIMES if int(round(t * args.fps)) <= steps}
    pred_trace = []

    t0 = time.time()
    for step in range(1, steps + 1):
        for j, s in enumerate(specs):
            if len(obs[j]) > 0:
                if s["model"] is not None:
                    batch, _ = s["model"].predict(obs[j], deterministic=True)
                    actions = [int(a) for a in np.atleast_1d(batch)]
                else:
                    actions = [C.rule_action("flee_lead", obs[j][i]) for i in range(len(obs[j]))]
            else:
                actions = []
            o, _r, term, trunc, info = envs[j].step(actions)
            obs[j] = np.asarray(o)

        # Shared predator trajectory: action-independent, must match across panels.
        p0 = envs[0].predator_pos
        for j, e in enumerate(envs[1:], 1):
            assert np.array_equal(p0, e.predator_pos), f"predator diverged at step {step} panel {j}"
            assert np.array_equal(envs[0].predator_vel, e.predator_vel), f"predator vel diverged at step {step} panel {j}"
        pred_trace.append(float(p0[0]))

        img = base.copy()
        d = ImageDraw.Draw(img)
        for j, e in enumerate(envs):
            draw_panel(img, d, j, e, copy, f_body)
        elapsed = step / float(args.fps)
        for j in range(3):
            cx = j * PANEL_W + PANEL_W // 2
            ctext(d, (cx, TIMER_Y + 14), f"{copy['elapsed_label']} {elapsed:.1f} s", f_body, C_TEXT)
        frame_path = run_dir / "frames" / f"f{step:04d}.png"
        img.save(frame_path)
        if step in posters_at:
            img.save(run_dir / "posters" / f"poster_{posters_at[step]:g}s.png")
        if step % 100 == 0:
            print(f"[render] step {step}/{steps}")

    render_wall = time.time() - t0
    n_frames = len(list((run_dir / "frames").glob("f*.png")))
    assert n_frames == steps, f"expected {steps} frames, found {n_frames}"

    # Contact sheet from posters plus two extra beats.
    sheet_frames = [f for f in [1, 100, 250, 400, 500] if f <= steps]
    thumbs = [Image.open(run_dir / "frames" / f"f{f:04d}.png") for f in sheet_frames]
    tw, th = 600, int(600 * H / W)
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), (255, 255, 255))
    for k, t in enumerate(thumbs):
        sheet.paste(t.resize((tw, th)), ((k % cols) * tw, (k // cols) * th))
    sheet.save(run_dir / "contact_sheet.png")

    mp4 = run_dir / "before_after.mp4"
    log = open(run_dir / "ffmpeg.log", "w")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(run_dir / "frames" / "f%04d.png"),
        "-c:v", "libx264", "-crf", "24", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log.write(proc.stdout + "\n" + proc.stderr)
    log.close()
    assert proc.returncode == 0, f"ffmpeg failed: {proc.stderr[-800:]}"

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate,codec_name,pix_fmt",
         "-show_entries", "format=duration", "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True,
    )
    probe_data = json.loads(probe.stdout)
    size_mb = mp4.stat().st_size / 1e6

    meta = {
        "kind": "before_after_video_meta",
        "scene_seed": int(seed),
        "scene_index": 0,
        "panels": ["old_ppo", "new_rep0", "rule_flee_lead"],
        "fps": args.fps, "steps": steps, "frames": n_frames,
        "expected_duration_sec": steps / args.fps,
        "size_mb": size_mb,
        "ffprobe": probe_data,
        "render_wall_sec": render_wall,
        "predator_trace_head": pred_trace[:5],
        "shared_initial_state": True,
        "shared_predator_trajectory": True,
    }
    (run_dir / "video_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)[:1200])
    print(f"[written] {mp4}  ({size_mb:.2f} MB)")
    if size_mb > 10.0:
        print("WARNING: mp4 exceeds 10 MB")


if __name__ == "__main__":
    main()
