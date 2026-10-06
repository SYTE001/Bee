"""Fase 3 verification: run the new visual + reframe stages on a real video.

Run:  python scripts/smoke_reframe.py

Builds a synthetic podcast-style video with two moving "speakers", runs the
visual analysis and per-clip reframing, and reports:
  * scene-cut detection
  * the visual score dimension per clip
  * the crop plan each clip received (layout, keyframes, ffmpeg filter)

This exercises the same code path the pipeline uses, without needing the API
server or a whisper model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.models.clip import Clip  # noqa: E402
from app.db.models.project import Project  # noqa: E402
from app.db.models.user import User  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.pipelines import video_pipeline as vp  # noqa: E402
from app.services import artifacts, reframe, visual_analysis  # noqa: E402
from app.services.storage import storage  # noqa: E402


def synth_two_speaker(path: Path) -> Path:
    """A 1280x720 video with two face-like patches taking turns moving.

    Built so the detector sees two "people": a left speaker and a right
    speaker, each active during their turn. Gives the auto-layout logic
    something realistic to choose split_screen for.
    """
    import cv2
    import numpy as np

    fps, duration = 15, 12.0
    n = int(fps * duration)
    tmpl = "data/smoke/spk_%03d.png"
    for i in range(n):
        t = i / fps
        bg = np.full((720, 1280, 3), 28, np.uint8)
        # Both speakers are on camera the whole time (a two-shot); each
        # *moves* during their own turn so the active-speaker heuristic has
        # something to latch onto.
        for side, active_from, active_to in ((0, 0.0, 6.0), (1, 6.0, 12.0)):
            base_x = 300 + side * 620
            x = base_x + (int(60 * (t - active_from)) if active_from <= t < active_to else 0)
            y = 280 + (int(24 * (t - active_from)) if active_from <= t < active_to else 0)
            colour = (190, 150, 120) if side == 0 else (150, 170, 190)
            cv2.ellipse(bg, (x, y), (90, 120), 0, 0, 360, colour, -1)
            cv2.ellipse(bg, (x - 30, y - 30), (12, 8), 0, 0, 360, (40, 30, 30), -1)
            cv2.ellipse(bg, (x + 30, y - 30), (12, 8), 0, 0, 360, (40, 30, 30), -1)
            cv2.ellipse(bg, (x, y + 25), (26, 13), 0, 0, 360, (60, 40, 40), -1)
        # A real hard cut at t=6: swap to a visibly different background.
        if t >= 6.0:
            bg = np.full((720, 1280, 3), 90, np.uint8)
            for side, active_from, active_to in ((0, 0.0, 6.0), (1, 6.0, 12.0)):
                base_x = 300 + side * 620
                x = base_x + int(60 * (t - active_from))
                y = 280 + int(24 * (t - active_from))
                colour = (190, 150, 120) if side == 0 else (150, 170, 190)
                cv2.ellipse(bg, (x, y), (90, 120), 0, 0, 360, colour, -1)
                cv2.ellipse(bg, (x - 30, y - 30), (12, 8), 0, 0, 360, (40, 30, 30), -1)
                cv2.ellipse(bg, (x + 30, y - 30), (12, 8), 0, 0, 360, (40, 30, 30), -1)
                cv2.ellipse(bg, (x, y + 25), (26, 13), 0, 0, 360, (60, 40, 40), -1)
        cv2.imwrite(tmpl % i, bg)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", tmpl,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    for i in range(n):
        Path(tmpl % i).unlink(missing_ok=True)
    return path


def main() -> None:
    tmp = Path("data/smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    src = synth_two_speaker(tmp / "two_speaker.mp4")
    print(f"built {src} ({src.stat().st_size} bytes)\n")

    # --- visual analysis -------------------------------------------------
    t0 = time.time()
    feats = visual_analysis.analyze_video(src)
    print(f"visual analysis: {time.time() - t0:.1f}s")
    print(f"  samples={len(feats.times)} cuts={feats.cuts}")
    print(f"  face_area mean={sum(feats.face_area) / max(len(feats.face_area), 1):.3f}")
    print(f"  motion   mean={sum(feats.motion) / max(len(feats.motion), 1):.3f}\n")

    # --- per-clip reframing, one clip per layout --------------------------
    print("=== crop plans per layout ===")
    for layout in ("auto", "single_focus", "split_screen", "fit_blur", "fill_zoom"):
        t0 = time.time()
        frames, det = reframe.analyze_faces(src, start=0.0, end=12.0)
        plan = reframe.build_crop_plan(
            frames, layout=layout, width=1280, height=720, detector=det
        )
        vf = reframe.crop_filter(plan)
        print(
            f"  {layout:13s} -> {plan.layout:13s} "
            f"crop={plan.crop_w:.0f}x{plan.crop_h:.0f} "
            f"keys={len(plan.samples)} static={plan.is_static} "
            f"({time.time() - t0:.1f}s, detector={det})"
        )
        print(f"      filter: {vf[:100]}{'...' if len(vf) > 100 else ''}")
        # render it to prove the filter is valid ffmpeg
        out = tmp / f"reframe_{layout}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-vf", f"{vf},{reframe.scale_filter(plan)}",
             "-t", "12", "-an", "-c:v", "libx264", "-crf", "20",
             "-pix_fmt", "yuv420p", str(out)],
            check=True, capture_output=True,
        )
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        )
        print(f"      rendered: {probe.stdout.strip()} ({out.stat().st_size} bytes)")

    # --- visual score per hypothetical clip window ------------------------
    print("\n=== visual score by window ===")
    for start, end in ((0, 6), (6, 12), (0, 12), (3, 9)):
        stats = visual_analysis.clip_stats(feats, start, end)
        print(
            f"  {start:2d}-{end:2d}s cuts={stats.cuts_in_clip} "
            f"density={stats.cut_density:.2f}/s face={stats.face_presence:.3f} "
            f"motion={stats.face_motion:.3f} -> visual={visual_analysis.visual_score(stats):.3f}"
        )


if __name__ == "__main__":
    main()
