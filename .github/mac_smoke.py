"""Smoke test for an installed AutoEdit on macOS (run by mac-test.yml).

Runs with the installed venv's Python.  Network access is blocked once the
imports are done, so the AI models must come from the install's bundled
``models`` folder, exactly as on a freelancer's Mac.
"""

import json
import os
import platform
import socket
import sys
import time
import urllib.request

machine = platform.machine()
print("python", sys.version.split()[0], "machine", machine, "macOS", platform.mac_ver()[0])

from autoedit import __version__, device  # noqa: E402  (sets the Metal fallback)
from autoedit.paths import configure_model_cache  # noqa: E402

print("autoedit", __version__, "models:", configure_model_cache())

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rawpy  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import webview  # noqa: E402
from PIL import Image  # noqa: E402

print("torch", torch.__version__, "torchvision", torchvision.__version__, "numpy", np.__version__,
      "opencv", cv2.__version__, "libraw", rawpy.libraw_version, "pywebview", getattr(webview, "__version__", "?"))

want = "mps" if machine == "arm64" else "cpu"
got = device.default_device()
print("device", got, "| detector", device.detector_device(), "|", device.describe())
LOCAL = bool(os.environ.get("SMOKE_LOCAL"))  # a dry run on a developer PC
assert LOCAL or got == want, f"expected {want}, got {got}"

# from here on nothing may be downloaded
_connect = socket.socket.connect


def _local_only(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise OSError(f"network blocked in the smoke test: {address}")
    return _connect(self, address)


socket.socket.connect = _local_only

rng = np.random.default_rng(0)
photo = Image.fromarray((rng.random((345, 518, 3)) * 255).astype("uint8"))

t = time.time()
from autoedit.features.embed import Embedder  # noqa: E402

emb = Embedder(allow_cpu=True, batch_size=2)
vec = emb.embed_batch([photo, photo.transpose(Image.FLIP_LEFT_RIGHT)])
assert vec.shape[0] == 2 and np.isfinite(vec).all(), vec.shape
print(f"DINOv2 embedding on {emb.device}: {vec.shape} in {time.time() - t:.1f}s")

t = time.time()
from autoedit.crop.config import CropConfig  # noqa: E402
from autoedit.crop.straighten import StraightenModel, _bins, _build_net  # noqa: E402

cfg = CropConfig()
bins = _bins(cfg.angle_range_deg, cfg.bin_deg)
model = StraightenModel(_build_net(len(bins)).to(got).eval(), bins, cfg, got)
pred = model.predict_images([photo])[0]
print(f"straighten model on {model.device}: {pred.angle:+.2f} deg (conf {pred.confidence:.2f}) in {time.time() - t:.1f}s")

t = time.time()
from autoedit.crop.detect import PersonDetector  # noqa: E402

det = PersonDetector(allow_cpu=True)
found = det.detect([photo])[0]
assert found.poses is not None
print(f"people finder on {det.device}: {len(found.people)} people in {time.time() - t:.1f}s")

from autoedit.crop.lines import estimate_tilt  # noqa: E402

print("line detector:", estimate_tilt(photo))

from autoedit.crop.strict import example_template, fit  # noqa: E402

f = fit(example_template(), [(500, 250, 560, 330)], 1200, 800)
assert f.ok, f.reason
print("strict crop fit ok")

from autoedit.ui.app import serve_in_thread  # noqa: E402

server, port = serve_in_thread()
base = f"http://127.0.0.1:{port}"
for _ in range(60):
    try:
        v = json.load(urllib.request.urlopen(base + "/api/version", timeout=2))
        break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit("the engine did not start")
dash = json.load(urllib.request.urlopen(base + "/api/dashboard", timeout=120))
tc = dash["checks"]["checks"]["torch"]
print("engine:", v, "| mps", tc.get("mps_available"), "| cuda", tc.get("cuda_available"))
assert LOCAL or v["installed"] is True, "install_dir() did not recognise the Mac install"
assert LOCAL or bool(tc.get("mps_available")) == (machine == "arm64")
page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
assert 'data-page="crop"' in page
print("SMOKE OK")
