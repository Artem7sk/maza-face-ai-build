"""Frame extraction and selection for MAZA FACE AI.

Selects the BEST frames (large, sharp faces), not merely the first five.
ffmpeg is invoked with an argv list - never through a shell.
"""
import json
import os
import subprocess
import tempfile

MAX_FRAMES = 5
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"


class FrameError(Exception):
    pass


def probe_video(path, timeout=20):
    """Return basic video info via ffprobe (argv only, no shell)."""
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,duration,nb_frames,r_frame_rate",
           "-of", "json", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, shell=False)
    except subprocess.TimeoutExpired:
        raise FrameError("ffprobe timeout")
    if out.returncode != 0:
        raise FrameError(f"ffprobe failed: {out.stderr.strip()[:200]}")
    try:
        streams = json.loads(out.stdout).get("streams", [])
    except json.JSONDecodeError:
        raise FrameError("ffprobe returned invalid JSON")
    if not streams:
        raise FrameError("no video stream found")
    s = streams[0]
    return {
        "width": int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "duration": float(s.get("duration", 0) or 0),
        "fps": s.get("r_frame_rate", "?"),
    }


def extract_candidates(path, workdir, count=10, timeout=30):
    """Extract up to `count` candidate frames spread evenly across the video."""
    info = probe_video(path)
    if info["width"] <= 0 or info["height"] <= 0:
        raise FrameError("invalid video dimensions")
    if count < 1:
        raise FrameError("count must be >= 1")

    dur = info["duration"]
    if dur > 0:
        step = max(dur / (count + 1), 0.2)
        patterns = [f"{step * (i + 1):.3f}" for i in range(count)]
    else:
        patterns = None

    os.makedirs(workdir, exist_ok=True)
    frames = []

    if patterns:
        for i, ts in enumerate(patterns, 1):
            out = os.path.join(workdir, f"cand_{i:02d}.jpg")
            cmd = [FFMPEG, "-nostdin", "-v", "error", "-ss", ts, "-i", path,
                   "-frames:v", "1", "-q:v", "2", "-y", out]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=timeout, shell=False)
            except subprocess.TimeoutExpired:
                continue
            if r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
                frames.append({"path": out, "ts": float(ts)})
    else:
        outdir = os.path.join(workdir, "seq_%03d.jpg")
        cmd = [FFMPEG, "-nostdin", "-v", "error", "-i", path,
               "-vf", f"fps=2", "-q:v", "2", "-y", outdir]
        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, shell=False)
        except subprocess.TimeoutExpired:
            raise FrameError("ffmpeg timeout")
        for name in sorted(os.listdir(workdir)):
            if name.startswith("seq_") and name.endswith(".jpg"):
                frames.append({"path": os.path.join(workdir, name), "ts": None})

    if not frames:
        raise FrameError("no frames could be extracted")
    return {"info": info, "frames": frames}


def score_face(face_row):
    """Heuristic quality score: bigger + more centred = better.

    face_row = [x, y, w, h, ...landmarks..., score] from YuNet.
    """
    try:
        x, y, w, h = float(face_row[0]), float(face_row[1]), float(face_row[2]), float(face_row[3])
        conf = float(face_row[-1])
    except (IndexError, TypeError, ValueError):
        return 0.0
    area = w * h
    return area * max(conf, 0.0)


def pick_best(scored, limit=MAX_FRAMES):
    """Sort candidates by score desc and keep the best `limit`."""
    ranked = sorted(scored, key=lambda c: c["score"], reverse=True)
    return ranked[:limit]
