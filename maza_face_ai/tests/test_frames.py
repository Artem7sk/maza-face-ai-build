#!/usr/bin/env python3
"""Unit tests for MAZA FACE AI frame handling and SFace call contract.

Runs WITHOUT OpenCV models: uses fakes to prove the call contract.
Guards against the exact regression that broke run 40691404641:
    AttributeError: 'tuple' object has no attribute 'shape'
"""
import sys
import unittest

sys.path.insert(0, "/app")

import main as m  # noqa: E402


class FakeArray:
    def __init__(self, shape):
        self.shape = shape


class TestFrameFormat(unittest.TestCase):
    def test_build_frames_returns_name_and_array_pairs(self):
        """build_frames must return (name, ndarray) tuples - never bare arrays."""
        import numpy as np
        frames = m.build_frames(np)

        self.assertIsInstance(frames, list)
        self.assertGreaterEqual(len(frames), 5, "expected at least 5 test frames")

        for item in frames:
            self.assertIsInstance(item, tuple, f"frame is {type(item).__name__}, expected tuple")
            self.assertEqual(len(item), 2, "frame tuple must be exactly (name, array)")
            name, img = item
            self.assertIsInstance(name, str)
            self.assertTrue(hasattr(img, "shape"), "frame image has no .shape")
            self.assertEqual(len(img.shape), 3, "frame image must be HxWxC")

    def test_validate_frames_rejects_bare_array(self):
        """A bare ndarray (the old bug) must be rejected, not silently accepted."""
        bad = [FakeArray((480, 640, 3))]  # element is the image itself, not (name, img)
        with self.assertRaises(Exception):
            m.validate_frames(bad)

    def test_validate_frames_rejects_wrong_tuple_arity(self):
        bad = [("name", FakeArray((480, 640, 3)), "extra")]
        with self.assertRaises(Exception):
            m.validate_frames(bad)

    def test_validate_frames_rejects_2d_array(self):
        bad = [("name", FakeArray((480, 640)))]
        with self.assertRaises(Exception):
            m.validate_frames(bad)

    def test_validate_frames_accepts_correct_format(self):
        good = [("name", FakeArray((480, 640, 3)))]
        out = m.validate_frames(good)
        self.assertEqual(len(out), 1)

    def test_input_size_reads_second_element(self):
        """setInputSize must get (width, height) from the array, not from the tuple."""
        frames = [("a", FakeArray((480, 640, 3)))]
        validated = m.validate_frames(frames)
        w, h = m.input_size_of(validated)
        self.assertEqual((w, h), (640, 480))


class FakeRecognizer:
    """Records the call contract for SFace."""
    def __init__(self):
        self.aligned_inputs = []
        self.feature_inputs = []
        self.match_args = []

    def alignCrop(self, img, face):
        self.aligned_inputs.append((id(img), id(face)))
        return ("ALIGNED", id(img), id(face))

    def feature(self, aligned):
        self.feature_inputs.append(aligned)
        return "FEATURE"

    def match(self, a, b, mode):
        self.match_args.append(mode)
        return 0.99


class TestSFaceContract(unittest.TestCase):
    def test_feature_receives_aligned_crop_not_raw_face(self):
        """SFace.feature() must receive alignCrop() output, never a raw detection row."""
        rec = FakeRecognizer()
        img = FakeArray((480, 640, 3))
        faces = [object()]

        feats, aligned_count = m.embed_faces(rec, img, faces)

        self.assertEqual(aligned_count, 1)
        self.assertTrue(rec.aligned_inputs, "alignCrop was never called")
        self.assertEqual(rec.feature_inputs, [("ALIGNED", id(img), id(faces[0]))],
                         "feature() got something other than alignCrop() output")
        self.assertEqual(feats, ["FEATURE"])

    def test_match_uses_fr_cosine_constant(self):
        """match() must be called with FaceRecognizerSF_FR_COSINE."""
        rec = FakeRecognizer()
        const = getattr(m, "SFR_MODE", None)
        self.assertEqual(const, "FR_COSINE",
                         "module must expose SFR_MODE='FR_COSINE' (cv2.FaceRecognizerSF_FR_COSINE)")
        mode = m.match_mode(rec, cv2_module=None, const_value=16)
        self.assertEqual(mode, 16)
        self.assertEqual(rec.match_args, [16])


if __name__ == "__main__":
    unittest.main(verbosity=2)
