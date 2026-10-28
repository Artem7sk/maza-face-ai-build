"""Path safety for MAZA FACE AI input handling.

Rules enforced here (no shell, ever):
  * input must live inside /media/face_test (realpath comparison)
  * '..' traversal is rejected before any filesystem access
  * symlinks are resolved; if the real target escapes the root -> reject
  * only regular files with allowed extensions are accepted
  * paths are never passed to a shell; ffmpeg/OpenCV get argv only
"""
import os
import stat

ALLOWED_ROOT = "/media/face_test"
ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".jpg", ".jpeg", ".png"}
MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB


class UnsafePathError(Exception):
    """Raised when a path violates the container's input policy."""


def _real(path):
    return os.path.realpath(os.path.abspath(path))


def resolve_input(path, root=ALLOWED_ROOT):
    """Validate and return the real absolute path of an allowed input file.

    Raises UnsafePathError with a machine-readable reason.
    """
    if not isinstance(path, str) or not path.strip():
        raise UnsafePathError("empty path")

    raw = path.strip()

    # 1. reject traversal in the RAW string before touching the filesystem
    parts = raw.replace("\\", "/").split("/")
    if ".." in parts:
        raise UnsafePathError("path traversal '..' is not allowed")

    # 2. must be absolute (Supervisor passes absolute container paths)
    if not os.path.isabs(raw):
        raise UnsafePathError("path must be absolute")

    real_root = _real(root)
    real_path = _real(raw)

    # 3. containment check on RESOLVED paths (defeats symlink escapes)
    try:
        common = os.path.commonpath([real_root, real_path])
    except ValueError:
        raise UnsafePathError("path outside allowed root")
    if common != real_root:
        raise UnsafePathError(f"path outside allowed root {real_root}")

    if real_path == real_root:
        # directory input is allowed (folder of JPG)
        if not os.path.isdir(real_path):
            raise UnsafePathError("not a directory")
        return real_path

    # 4. reject symlinked final target pointing elsewhere (double check)
    if os.path.islink(raw):
        link_target = _real(raw)
        if os.path.commonpath([real_root, link_target]) != real_root:
            raise UnsafePathError("symlink escapes allowed root")

    if not os.path.exists(real_path):
        raise UnsafePathError("path does not exist")

    # 5. regular file only (no devices, sockets, fifos)
    st = os.lstat(real_path)
    if not stat.S_ISREG(st.st_mode):
        raise UnsafePathError("not a regular file")

    ext = os.path.splitext(real_path)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise UnsafePathError(f"extension {ext!r} not allowed")

    if st.st_size <= 0:
        raise UnsafePathError("file is empty")
    if st.st_size > MAX_FILE_BYTES:
        raise UnsafePathError(f"file too large: {st.st_size} bytes")

    return real_path


def list_inputs(path, root=ALLOWED_ROOT):
    """Return a sorted list of allowed files for a validated input path."""
    real = resolve_input(path, root)
    if os.path.isfile(real):
        return [real]
    out = []
    for name in sorted(os.listdir(real)):
        child = os.path.join(real, name)
        try:
            out.append(resolve_input(child, root))
        except UnsafePathError:
            continue  # silently skip disallowed entries
    return out
