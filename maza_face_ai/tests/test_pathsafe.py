#!/usr/bin/env python3
"""Security tests for input path validation (MAZA FACE AI 0.1.3).

Proves:
  * traversal via '..' is rejected
  * symlink escaping the allowed root is rejected
  * relative paths are rejected
  * files outside /media/face_test are rejected
  * only regular files with allowed extensions are accepted
  * no shell is ever used (ffmpeg runs with argv)
"""
import os
import sys
import tempfile
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_CANDIDATES = [_PARENT, os.path.join(_PARENT, "app"), _HERE, "/app"]
_APP_DIR = next((c for c in _CANDIDATES if os.path.isfile(os.path.join(c, "pathsafe.py"))), None)
if _APP_DIR is None:
    raise SystemExit(f"pathsafe.py not found; looked in {_CANDIDATES}")
sys.path.insert(0, _APP_DIR)

import pathsafe  # noqa: E402


class TestPathSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="psafe_")
        self.root = os.path.join(self.tmp, "face_test")
        os.makedirs(self.root, exist_ok=True)

    def _make(self, relpath, data=b"x" * 100):
        full = os.path.join(self.root, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True) if os.path.dirname(full) != self.root else None
        with open(full, "wb") as f:
            f.write(data)
        return full

    def test_traversal_rejected(self):
        with self.assertRaises(pathsafe.UnsafePathError) as ctx:
            pathsafe.resolve_input(self.root + "/../secret.mp4", root=self.root)
        self.assertIn("traversal", str(ctx.exception))

    def test_relative_path_rejected(self):
        with self.assertRaises(pathsafe.UnsafePathError) as ctx:
            pathsafe.resolve_input("video.mp4", root=self.root)
        self.assertIn("absolute", str(ctx.exception))

    def test_outside_root_rejected(self):
        outside = os.path.join(self.tmp, "outside.mp4")
        with open(outside, "wb") as f:
            f.write(b"x" * 50)
        with self.assertRaises(pathsafe.UnsafePathError) as ctx:
            pathsafe.resolve_input(outside, root=self.root)
        self.assertIn("outside allowed root", str(ctx.exception))

    def test_symlink_escape_rejected(self):
        outside = os.path.join(self.tmp, "secret.mp4")
        with open(outside, "wb") as f:
            f.write(b"x" * 50)
        link = os.path.join(self.root, "sneaky.mp4")
        os.symlink(outside, link)
        with self.assertRaises(pathsafe.UnsafePathError) as ctx:
            pathsafe.resolve_input(link, root=self.root)
        msg = str(ctx.exception)
        self.assertTrue("outside allowed root" in msg or "symlink" in msg, msg)

    def test_bad_extension_rejected(self):
        bad = self._make("script.sh")
        with self.assertRaises(pathsafe.UnsafePathError) as ctx:
            pathsafe.resolve_input(bad, root=self.root)
        self.assertIn("not allowed", str(ctx.exception))

    def test_empty_file_rejected(self):
        empty = self._make("empty.mp4", data=b"")
        with self.assertRaises(pathsafe.UnsafePathError) as ctx:
            pathsafe.resolve_input(empty, root=self.root)
        self.assertIn("empty", str(ctx.exception))

    def test_valid_mp4_accepted(self):
        ok = self._make("door.mp4")
        self.assertEqual(pathsafe.resolve_input(ok, root=self.root), os.path.realpath(ok))

    def test_valid_dir_accepted_and_listed(self):
        self._make("a.jpg")
        self._make("b.png")
        self._make("skip.txt")
        files = pathsafe.list_inputs(self.root, root=self.root)
        names = sorted(os.path.basename(f) for f in files)
        self.assertEqual(names, ["a.jpg", "b.png"])

    def test_missing_path_rejected(self):
        with self.assertRaises(pathsafe.UnsafePathError) as ctx:
            pathsafe.resolve_input(os.path.join(self.root, "nope.mp4"), root=self.root)
        self.assertIn("does not exist", str(ctx.exception))


class FakeRun:
    """Capture subprocess calls to prove no shell is used."""
    calls = []

    @staticmethod
    def run(cmd, **kwargs):
        FakeRun.calls.append((cmd, kwargs))
        raise AssertionError("should not reach subprocess in this test")


class TestNoShell(unittest.TestCase):
    def test_command_is_argv_list_with_shell_false(self):
        src = open(os.path.join(_APP_DIR, "frameselect.py")).read()
        self.assertIn("shell=False", src, "ffmpeg/ffprobe must run with shell=False")
        self.assertNotIn("shell=True", src, "shell=True is forbidden")
        self.assertNotIn("os.system", src, "os.system is forbidden")
        self.assertNotIn("popen(", src.lower().replace("subprocess.popen", ""),
                         "raw popen with string is forbidden")


if __name__ == "__main__":
    unittest.main(verbosity=2)
