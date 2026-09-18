"""Download the ArcFace recognition model onto the robot.

Run once during setup (scripts/setup_remote.bat does this). The model is a
13 MB binary, so it is fetched here rather than committed to the repo.

  w600k_mbf.onnx  -- ArcFace recognition, 512-dim embeddings, from the
  insightface buffalo_s pack. Chosen over buffalo_l's w600k_r50: measured on
  this robot's aarch64 CPU, r50 takes ~1460 ms per embedding versus ~117 ms
  for mbf, and identification runs on the camera thread.

Safe to re-run: it skips the download if the model is already present.
"""

import os
import sys
import urllib.request
import zipfile

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
MEMBER = "w600k_mbf.onnx"
DEST_DIR = "models"
DEST = os.path.join(DEST_DIR, MEMBER)
EXPECTED_MIN_BYTES = 10 * 1024 * 1024


def main() -> int:
    if os.path.exists(DEST) and os.path.getsize(DEST) >= EXPECTED_MIN_BYTES:
        print(f"OK: {DEST} already present ({os.path.getsize(DEST)} bytes).")
        return 0

    os.makedirs(DEST_DIR, exist_ok=True)
    tmp_zip = os.path.join(DEST_DIR, "_buffalo_s.zip")

    print(f"Downloading {PACK_URL} ...")
    try:
        urllib.request.urlopen  # noqa: B018 - explicit about what we use
        with urllib.request.urlopen(PACK_URL, timeout=120) as resp, open(tmp_zip, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        print(f"FAIL: could not download model pack: {e}")
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        return 1

    try:
        with zipfile.ZipFile(tmp_zip) as z:
            with z.open(MEMBER) as src, open(DEST, "wb") as out:
                out.write(src.read())
    except Exception as e:
        print(f"FAIL: could not extract {MEMBER}: {e}")
        return 1
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

    size = os.path.getsize(DEST)
    if size < EXPECTED_MIN_BYTES:
        print(f"FAIL: {DEST} is only {size} bytes - download looks truncated.")
        return 1

    print(f"OK: wrote {DEST} ({size} bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
