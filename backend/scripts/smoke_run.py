"""Manual smoke run: full pipeline through the live API.

Run:  python scripts/smoke_run.py
Produces a synthetic 2-minute video, uploads it, and prints the ranked clips.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8000"


def synth(path: Path, speech_wav: Path | None = None) -> Path:
    """Build a test video: black canvas plus optional real speech audio."""
    if speech_wav and speech_wav.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi",
                "-i", "color=c=black:s=1280x720:d=180:r=30",
                "-i", str(speech_wav),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                "-map", "0:v:0", "-map", "1:a:0",
                str(path),
            ],
            check=True,
        )
        return path
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi",
            "-i", "color=c=black:s=1280x720:d=120:r=30",
            "-f", "lavfi",
            "-i", "sine=frequency=200:duration=120",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def main() -> None:
    import httpx

    tmp = Path("data/smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    speech = tmp / "speech.wav"
    video = synth(tmp / "podcast.mp4", speech_wav=speech)

    client = httpx.Client(base_url=BASE, timeout=600)
    email = f"smoke-{int(time.time())}@example.com"
    r = client.post("/api/auth/register", json={"email": email, "password": "password123"})
    r.raise_for_status()
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"registered {email}")

    r = client.post(
        "/api/projects",
        json={
            "source_type": "upload",
            "source_name": "smoke-podcast.mp4",
            "language": "en",
            "target_dur_min": 15,
            "target_dur_max": 60,
            "target_clip_count": 5,
        },
        headers=headers,
    )
    r.raise_for_status()
    project_id = r.json()["id"]

    with open(video, "rb") as fh:
        r = client.post(
            f"/api/projects/{project_id}/upload",
            files={"file": ("podcast.mp4", fh, "video/mp4")},
            data={"auto_start": "true"},
            headers=headers,
        )
    r.raise_for_status()
    print(f"uploaded, pipeline started: {project_id}")

    # poll
    for _ in range(120):
        r = client.get(f"/api/projects/{project_id}", headers=headers)
        r.raise_for_status()
        p = r.json()
        print(f"  [{p['status']}] {p['stage_message']} ({p['stage_progress'] * 100:.0f}%)")
        if p["status"] in ("completed", "failed"):
            break
        time.sleep(3)

    if p["status"] == "failed":
        print("PIPELINE FAILED:", p.get("error"))
        sys.exit(1)

    r = client.get(f"/api/projects/{project_id}/clips", headers=headers)
    r.raise_for_status()
    clips = r.json()
    print(f"\n=== {len(clips)} clips ===")
    for c in clips:
        print(
            f"#{c['rank']} score={c['viral_score']} [{c['category_primary']}/{c['category_secondary']}] "
            f"{c['start']:.1f}-{c['end']:.1f}s ({c['duration']:.1f}s)"
        )
        print(f"   title: {c['title']}")
        print(f"   tags:  {c['hashtags']}")
        print(f"   why:   {(c['reasons'] or ['?'])[0]}")


if __name__ == "__main__":
    main()
