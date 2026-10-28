#!/usr/bin/env python3
"""MAZA FACE AI - YuNet (detect) + SFace (compare), on-demand only, local."""
import os, sys, json, time, resource, signal

MAX_RUNTIME = int(os.environ.get("MAZA_MAX_RUNTIME", "60"))
MEM_LIMIT = 512 * 1024 * 1024

def apply_limits():
    """Apply hard self-limits: address space + single CPU core."""
    ok = {}
    # RLIMIT_AS = 512 MB virtual address space
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
        ok["rlimit_as"] = resource.getrlimit(resource.RLIMIT_AS)[0]
    except Exception as e:
        ok["rlimit_as"] = f"FAILED: {e}"
    # pin to one core
    try:
        os.sched_setaffinity(0, {0})
        ok["affinity"] = sorted(os.sched_getaffinity(0))
    except Exception as e:
        ok["affinity"] = f"FAILED: {e}"
    # thread caps for BLAS/OpenMP
    for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    return ok

def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return -1

def cur_rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1

def watchdog(signum, frame):
    print(json.dumps({"event":"timeout","max_runtime":MAX_RUNTIME}), flush=True)
    sys.exit(124)

def main():
    t_start = time.time()
    limits = apply_limits()
    print(json.dumps({"event":"start","limits":limits,"max_runtime":MAX_RUNTIME}), flush=True)

    signal.signal(signal.SIGALRM, watchdog)
    signal.alarm(MAX_RUNTIME)

    try:
        import cv2
        import numpy as np
    except Exception as e:
        print(json.dumps({"event":"error","stage":"import","error":str(e)}), flush=True)
        sys.exit(2)

    cv2.setNumThreads(1)
    print(json.dumps({"event":"opencv","version":cv2.__version__,
                      "threads":cv2.getNumThreads(),
                      "rss_kb":cur_rss_kb()}), flush=True)

    data_dir = "/data"
    os.makedirs(data_dir, exist_ok=True)
    models = "/app/models"
    yunet_path = os.path.join(models, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(models, "face_recognition_sface_2021dec.onnx")

    missing = [p for p in (yunet_path, sface_path) if not os.path.isfile(p)]
    if missing:
        print(json.dumps({"event":"error","stage":"models","missing":missing}), flush=True)
        sys.exit(3)

    try:
        det = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.9, 0.3, 5000)
        rec = cv2.FaceRecognizerSF.create(sface_path, "")
    except Exception as e:
        print(json.dumps({"event":"error","stage":"model_load","error":str(e),
                          "rss_kb":cur_rss_kb()}), flush=True)
        sys.exit(4)

    print(json.dumps({"event":"models_loaded","rss_kb":cur_rss_kb(),
                      "vmhwm_kb":rss_kb(),
                      "elapsed_s":round(time.time()-t_start,3)}), flush=True)

    # smoke test on a synthetic frame: proves pipeline runs without external images
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    det.setInputSize((640, 480))
    t0 = time.perf_counter()
    _, faces = det.detect(img)
    detect_ms = (time.perf_counter() - t0) * 1000
    print(json.dumps({"event":"smoke","detect_ms":round(detect_ms,2),
                      "faces":0 if faces is None else len(faces),
                      "rss_kb":cur_rss_kb(),"vmhwm_kb":rss_kb()}), flush=True)

    print(json.dumps({"event":"done","total_s":round(time.time()-t_start,3),
                      "vmhwm_kb":rss_kb()}), flush=True)

if __name__ == "__main__":
    main()
