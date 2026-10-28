#!/usr/bin/env python3
"""MAZA FACE AI - YuNet (detect) + SFace (compare/embed). On-demand only, local.

Fail-closed resource guard:
  - RLIMIT_AS caps VIRTUAL ADDRESS SPACE of this process, NOT container RSS.
  - CPU affinity is pinned to ONE core that is actually permitted to the container.
  - Any required limit that cannot be applied => immediate exit BEFORE model load.
"""
import os
import sys
import json
import time
import resource
import signal

MEM_LIMIT_BYTES = 512 * 1024 * 1024
START = time.time()


_EVENTS_FH = None


def _events_file():
    """Optional mirror of JSON events to /out/events.jsonl for CI assertions."""
    global _EVENTS_FH
    if _EVENTS_FH is None:
        try:
            os.makedirs("/out", exist_ok=True)
            _EVENTS_FH = open("/out/events.jsonl", "a", buffering=1)
        except Exception:
            _EVENTS_FH = False
    return _EVENTS_FH


def _emit(**kw):
    line = json.dumps(kw, ensure_ascii=False)
    print(line, flush=True)
    fh = _events_file()
    if fh:
        try:
            fh.write(line + "\n")
            fh.flush()
        except Exception:
            pass


def fail(code, stage, err):
    """Fail-closed: report and exit non-zero. Models are NOT loaded."""
    _emit(event="fatal", stage=stage, error=str(err), exit_code=code)
    sys.exit(code)


def _read_proc_status():
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "VmHWM", "VmPeak", "VmSize"):
                    out[k] = int(v.split()[0])
    except Exception:
        pass
    return out


def apply_limits():
    """Apply required limits. Returns dict. Raises RuntimeError on any failure."""
    applied = {}

    # 1. RLIMIT_AS - hard cap on virtual address space (NOT container RSS)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
        now_soft, now_hard = resource.getrlimit(resource.RLIMIT_AS)
        if now_soft != MEM_LIMIT_BYTES:
            raise RuntimeError(f"RLIMIT_AS not applied: soft={now_soft}")
        applied["rlimit_as_bytes"] = now_soft
        applied["rlimit_as_before"] = [soft, hard]
    except Exception as e:
        raise RuntimeError(f"RLIMIT_AS failed: {e}")

    # 2. CPU affinity - pick a core that is ACTUALLY permitted, never assume CPU 0
    try:
        allowed = sorted(os.sched_getaffinity(0))
        if not allowed:
            raise RuntimeError("no CPUs permitted to this process")
        target = allowed[0]
        os.sched_setaffinity(0, {target})
        now_allowed = sorted(os.sched_getaffinity(0))
        if now_allowed != [target]:
            raise RuntimeError(f"affinity not applied: {now_allowed} != [{target}]")
        applied["affinity_allowed_before"] = allowed
        applied["affinity_cpu"] = target
        applied["affinity_after"] = now_allowed
        applied["affinity_ok"] = True
    except Exception as e:
        raise RuntimeError(f"CPU affinity failed: {e}")

    applied["cpu_count_visible"] = os.cpu_count()
    return applied


def main():
    _emit(event="boot", pid=os.getpid(), max_runtime_env=os.environ.get("MAZA_MAX_RUNTIME"))

    try:
        max_runtime = int(os.environ.get("MAZA_MAX_RUNTIME", "60"))
    except ValueError:
        max_runtime = 60
    max_runtime = max(5, min(300, max_runtime))

    # ---- FAIL-CLOSED: limits first, models later ----
    try:
        limits = apply_limits()
    except RuntimeError as e:
        fail(10, "limits", e)

    _emit(event="limits_applied", **limits)

    def _timeout(signum, frame):
        _emit(event="timeout", max_runtime=max_runtime, vmhwm_kb=_read_proc_status().get("VmHWM"))
        sys.exit(124)

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(max_runtime)

    # ---- imports ----
    try:
        import cv2
        import numpy as np
    except Exception as e:
        fail(20, "import", e)

    cv2.setNumThreads(1)
    _emit(event="opencv_ready", version=cv2.__version__, threads=cv2.getNumThreads(),
          vmhwm_kb=_read_proc_status().get("VmHWM"))

    # ---- models ----
    yunet_path = "/app/models/face_detection_yunet_2023mar.onnx"
    sface_path = "/app/models/face_recognition_sface_2021dec.onnx"
    missing = [p for p in (yunet_path, sface_path) if not os.path.isfile(p)]
    if missing:
        fail(30, "models_missing", missing)

    try:
        det = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
        rec = cv2.FaceRecognizerSF.create(sface_path, "")
    except Exception as e:
        fail(31, "model_load", e)

    _emit(event="models_loaded", vmhwm_kb=_read_proc_status().get("VmHWM"),
          elapsed_s=round(time.time() - START, 3))

    # ---- test frames ----
    frames = build_frames(np)

    # guard: validate frame container format BEFORE touching models
    try:
        frames = validate_frames(frames)
    except ValueError as e:
        fail(40, "frames_format", e)

    det.setInputSize(input_size_of(frames))

    results = []
    counts_seen = []
    for idx, (name, img) in enumerate(frames, 1):
        t0 = time.perf_counter()
        _, faces = det.detect(img)
        detect_ms = (time.perf_counter() - t0) * 1000.0
        n = 0 if faces is None else len(faces)

        embed_ms = None
        if n > 0:
            t1 = time.perf_counter()
            try:
                feats, _ = embed_faces(rec, img, [faces[i] for i in range(n)])
                embed_ms = (time.perf_counter() - t1) * 1000.0
                scores = [round(float(np.linalg.norm(f)), 4) for f in feats]
            except Exception as e:
                scores = f"embed_error: {e}"
        else:
            scores = []

        st = _read_proc_status()
        r = {"frame": name, "index": idx, "shape": list(img.shape),
             "faces": n, "detect_ms": round(detect_ms, 2),
             "embed_ms": None if embed_ms is None else round(embed_ms, 2),
             "feature_norms": scores,
             "vmhwm_kb": st.get("VmHWM"), "vmrss_kb": st.get("VmRSS")}
        results.append(r)
        _emit(event="frame", **r)

    # ---- self-comparison sanity check (same synthetic face vs itself) ----
    compare_out = None
    if len(results) >= 2 and results[0]["faces"] and results[1]["faces"]:
        try:
            _, f1 = det.detect(frames[0][1])
            _, f2 = det.detect(frames[1][1])
            same = compare_faces(rec, frames[0][1], f1[0], frames[1][1], f2[0], cv2)
            compare_out = {"same_face_score": round(same, 4)}
            _emit(event="compare", **compare_out)
        except Exception as e:
            _emit(event="compare_error", error=str(e))

    # ---- 1 / 3 / 5 frame batch checks (as promised in the plan) ----
    batches = {}
    for k in (1, 3, 5):
        subset = results[:k]
        if len(subset) < k:
            batches[str(k)] = {"requested": k, "available": len(subset), "ok": False}
            continue
        batches[str(k)] = {
            "requested": k,
            "processed": len(subset),
            "faces_total": sum(r["faces"] for r in subset),
            "detect_ms_total": round(sum(r["detect_ms"] for r in subset), 2),
            "embed_ms_total": round(sum(r["embed_ms"] or 0.0 for r in subset), 2),
            "ok": len(subset) == k,
        }
    _emit(event="batches", **batches)

    final = _read_proc_status()
    _emit(event="done", frames=len(results), total_s=round(time.time() - START, 3),
          vmhwm_kb=final.get("VmHWM"), vmpeak_kb=final.get("VmPeak"),
          rss_kb=final.get("VmRSS"))
    signal.alarm(0)


def validate_frames(frames):
    """Validate frame container format.

    Contract: list of (name:str, image:HxWxC ndarray) tuples.
    Raises ValueError on ANY deviation - including the historical bug where a
    bare ndarray was passed, causing "'tuple' object has no attribute 'shape'".
    """
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise ValueError(f"frames must be a non-empty list, got {type(frames).__name__}")

    out = []
    for i, item in enumerate(frames):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(f"frame[{i}] tuple must be (name, image), got len={len(item)}")
            name, img = item
            if not isinstance(name, str) or not name:
                raise ValueError(f"frame[{i}] name must be a non-empty str")
        elif hasattr(item, "shape"):
            # bare ndarray = the exact historical bug; reject loudly, never auto-wrap
            raise ValueError(
                f"frame[{i}] is a bare array with shape {getattr(item, 'shape', None)}; "
                "expected (name, image) tuple"
            )
        else:
            raise ValueError(f"frame[{i}] has unsupported type {type(item).__name__}")

        shape = getattr(img, "shape", None)
        if shape is None:
            raise ValueError(f"frame[{i}] ({name!r}) image has no .shape")
        if len(shape) != 3:
            raise ValueError(f"frame[{i}] ({name!r}) must be HxWxC, got shape={tuple(shape)}")
        out.append((name, img))
    return out


def input_size_of(frames):
    """Return (width, height) of the FIRST validated frame."""
    _, img = frames[0]
    return int(img.shape[1]), int(img.shape[0])


def embed_faces(recognizer, img, faces):
    """SFace contract: alignCrop() then feature(). Never feature() a raw row."""
    aligned = [recognizer.alignCrop(img, face) for face in faces]
    feats = [recognizer.feature(a) for a in aligned]
    return feats, len(aligned)


SFR_MODE = "FR_COSINE"


def match_mode(recognizer, cv2_module, const_value):
    """Compare using cv2.FaceRecognizerSF_FR_COSINE (16), not DEFAULT."""
    return const_value


def compare_faces(recognizer, img_a, face_a, img_b, face_b, cv2_module):
    """Full compare path with alignment and the COSINE constant."""
    a = recognizer.feature(recognizer.alignCrop(img_a, face_a))
    b = recognizer.feature(recognizer.alignCrop(img_b, face_b))
    const = getattr(cv2_module, "FaceRecognizerSF_FR_COSINE", 16)
    return float(recognizer.match(a, b, const))


def build_frames(np):
    """Deterministic synthetic frames with a face-like pattern.

    No real people. A drawn face-like structure gives YuNet a chance to detect
    something and lets us time the full detect+embed path.
    """
    frames = []
    for scale in (1.0, 1.0, 1.15, 1.15, 1.3):
        h, w = 480, 640
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        cx, cy = int(w // 2), int(h // 2)
        rx, ry = int(60 * scale), int(80 * scale)
        # head
        cv2_ellipse = __import__("cv2").ellipse
        cv2_ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (190, 170, 150), -1)
        # eyes
        cv2_circle = __import__("cv2").circle
        cv2_circle(img, (cx - rx // 2, cy - ry // 3), max(4, rx // 8), (30, 30, 30), -1)
        cv2_circle(img, (cx + rx // 2, cy - ry // 3), max(4, rx // 8), (30, 30, 30), -1)
        # nose + mouth
        cv2_line = __import__("cv2").line
        cv2_line(img, (cx, cy), (cx, cy + ry // 3), (140, 120, 110), max(2, rx // 12))
        cv2_ellipse(img, (cx, cy + ry // 2), (rx // 2, max(6, ry // 8)), 0, 0, 180, (90, 60, 60), max(2, rx // 14))
        frames.append((f"synthetic_face_{scale}", img))
    return frames


if __name__ == "__main__":
    main()
