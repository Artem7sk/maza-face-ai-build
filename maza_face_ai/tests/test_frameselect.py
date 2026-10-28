#!/usr/bin/env python3
"""Frame extraction tests against the bundled synthetic MP4 (no real footage)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_CANDIDATES = [_PARENT, os.path.join(_PARENT, "app"), _HERE, "/app"]
_APP_DIR = next((c for c in _CANDIDATES if os.path.isfile(os.path.join(c, "frameselect.py"))), None)
if _APP_DIR is None:
    raise SystemExit(f"frameselect.py not found; looked in {_CANDIDATES}")
sys.path.insert(0, _APP_DIR)

import frameselect  # noqa: E402


def _sample_mp4():
    for base in (os.path.dirname(_PARENT), _PARENT, "/workspace"):
        p = os.path.join(base, "testdata", "sample_10s.mp4")
        if os.path.isfile(p):
            return p
    return None


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg/ffprobe not installed")
class TestFrameExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = _sample_mp4()
        if not cls.video:
            raise unittest.SkipTest("testdata/sample_10s.mp4 not found")

    def test_probe_reports_dimensions(self):
        info = frameselect.probe_video(self.video)
        self.assertEqual(info["width"], 640)
        self.assertEqual(info["height"], 480)
        self.assertGreater(info["duration"], 5.0)

    def test_extracts_candidates(self):
        work = tempfile.mkdtemp(prefix="fs_")
        try:
            ex = frameselect.extract_candidates(self.video, work, count=10)
            self.assertGreaterEqual(len(ex["frames"]), 5)
            for f in ex["frames"]:
                self.assertTrue(os.path.isfile(f["path"]))
                self.assertGreater(os.path.getsize(f["path"]), 0)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_pick_best_limits_to_five(self):
        scored = [{"i": i, "score": float(i)} for i in range(12)]
        best = frameselect.pick_best(scored, limit=5)
        self.assertEqual(len(best), 5)
        self.assertEqual([b["i"] for b in best], [11, 10, 9, 8, 7])

    def test_score_prefers_larger_face(self):
        small = frameselect.score_face([0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9])
        big = frameselect.score_face([0, 0, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9])
        self.assertGreater(big, small)

    def test_broken_file_raises(self):
        work = tempfile.mkdtemp(prefix="fs_")
        try:
            bad = os.path.join(work, "broken.mp4")
            with open(bad, "wb") as f:
                f.write(b"not a video at all")
            with self.assertRaises(frameselect.FrameError):
                frameselect.probe_video(bad)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_no_shell_in_frameselect(self):
        src = open(os.path.join(_APP_DIR, "frameselect.py")).read()
        self.assertIn("shell=False", src)
        self.assertNotIn("shell=True", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
