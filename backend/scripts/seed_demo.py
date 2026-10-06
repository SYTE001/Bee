"""Seed a demo project so the Fase 5 UI can be exercised in a browser.

Creates a user, a completed project with three ranked clips, renders two of
them for real (ffmpeg), and prints the login email + project URL.

Run:  python scripts/seed_demo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.security import hash_password  # noqa: E402
from app.db.models.clip import Clip  # noqa: E402
from app.db.models.project import Project, ProjectStatus, SourceType  # noqa: E402
from app.db.models.user import User  # noqa: E402
from app.db.session import SessionLocal, create_all  # noqa: E402
from app.pipelines import video_pipeline as vp  # noqa: E402
from app.services import artifacts, render  # noqa: E402
from app.services.storage import storage  # noqa: E402

EMAIL = "demo@example.com"
PASSWORD = "demodemo"


def synth_source(path: Path) -> Path:
    """A 12s 1280x720 talking-head-ish clip with audio."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=s=1280x720:d=12:r=30",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def main() -> None:
    create_all()
    db = SessionLocal()

    user = db.query(User).filter(User.email == EMAIL).first()
    if user is None:
        user = User(email=EMAIL, hashed_password=hash_password(PASSWORD), is_active=True)
        db.add(user)
        db.commit()
    print(f"user: {EMAIL} / {PASSWORD}")

    # fresh project every run
    old = db.query(Project).filter(Project.user_id == user.id).all()
    for p in old:
        db.delete(p)
    db.commit()

    project = Project(
        user_id=user.id,
        source_type=SourceType.UPLOAD,
        source_name="demo-podcast.mp4",
        status=ProjectStatus.COMPLETED,
        stage_message="done",
        duration=12.0,
        width=1280,
        height=720,
        fps=30.0,
        detected_language="en",
        num_speakers=1,
        target_dur_min=3,
        target_dur_max=8,
        target_clip_count=3,
    )
    db.add(project)
    db.commit()

    pdir = storage.project_dir(project.id)
    pdir.mkdir(parents=True, exist_ok=True)
    target = pdir / "source.mp4"
    synth_source(target)

    store = artifacts.ArtifactStore(db, project.id)
    store.put(
        vp.ARTIFACT_SOURCE,
        str(target),
        {"duration": 12.0, "width": 1280, "height": 720, "fps": 30.0,
         "has_video": True, "has_audio": True},
    )

    samples = [
        {
            "rank": 1, "start": 0.5, "end": 6.5, "score": 92.4,
            "category": "insight", "title": "The consistency secret nobody talks about",
            "reasons": ["Strong hook in the first 2 seconds", "Standalone quote — works without context"],
            "words": ["the", "secret", "is", "consistency", "over", "talent", "and", "the", "numbers", "back", "it", "up"],
            "style": "hormozi", "layout": "single_focus",
        },
        {
            "rank": 2, "start": 4.0, "end": 9.5, "score": 78.1,
            "category": "funny", "title": "Wait, that actually works?",
            "reasons": ["Audible laughter peak", "Reaction beat lands at the cut"],
            "words": ["wait", "really", "that", "is", "honestly", "shocking", "to", "hear"],
            "style": "bounce_pop", "layout": "fit_blur",
        },
        {
            "rank": 3, "start": 8.0, "end": 11.5, "score": 64.8,
            "category": "actionable", "title": "Three steps you can use today",
            "reasons": ["Clear numbered list structure", "Ends on a CTA"],
            "words": ["first", "audit", "your", "time", "for", "one", "week"],
            "style": "classic_bold_white", "layout": "auto",
        },
    ]

    for s in samples:
        span = s["end"] - s["start"]
        per = span / len(s["words"])
        words = [
            {"word": w, "start": round(s["start"] + i * per, 3),
             "end": round(s["start"] + (i + 1) * per, 3), "speaker": "SPEAKER_00"}
            for i, w in enumerate(s["words"])
        ]
        clip = Clip(
            project_id=project.id,
            rank=s["rank"],
            start=s["start"], end=s["end"], duration=span,
            viral_score=s["score"],
            category_primary=s["category"],
            scores={"hook": s["score"] / 100, "emotion": 0.6, "standalone": 0.8},
            reasons=s["reasons"],
            title=s["title"],
            description=s["title"],
            hashtags="#podcast #viral",
            transcript=" ".join(s["words"]),
            words=words,
            subtitle_style=s["style"],
            layout_mode=s["layout"],
            target_fps=30,
        )
        db.add(clip)
    db.commit()

    print(f"project: http://localhost:5173/projects/{project.id}")
    print("rendering clips 1-2 (real ffmpeg)…")
    rendered, failed = render.render_project_clips(db, project, store)
    print(f"rendered={rendered} failed={failed}")

    # clip 3 stays unrendered so the UI can show the "Render" button
    pending = db.query(Clip).filter(Clip.project_id == project.id, Clip.rendered == False).count()
    print(f"pending (intentionally unrendered): {pending}")
    db.close()


if __name__ == "__main__":
    main()
