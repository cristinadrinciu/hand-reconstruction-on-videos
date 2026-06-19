"""
Stub launcher for WiLoR's ORIGINAL demo.py on the headless cluster.

WiLoR's demo renders with pyrender (EGL/OSMesa unusable on the GPU nodes), and
the model's __init__ also builds a pyrender OffscreenRenderer. This launcher:
  1. injects a fake `pyrender` module so nothing pyrender-related touches the GPU;
  2. neutralizes Renderer.render_rgba_multiple (the overlay step is 100% pyrender)
     so the demo runs to completion instead of crashing on every frame;
  3. runs the UNMODIFIED demo.py with its normal argparse args.

The reconstruction (detector + WiLoR model + MANO output) is IDENTICAL to the
original demo; only the pyrender overlay is skipped. Use --save_mesh to dump the
per-hand meshes (.obj, via trimesh -> survives the stub).

Usage (from WiLoR/WiLoR, in a WiLoR env on a GPU node):
    python run_demo_stub_wilor.py \
        --img_folder <frames_dir> --out_folder <out_dir> --save_mesh
"""
import sys
from types import ModuleType

import numpy as np


# --- 1. fake pyrender (no-op) ---
class _FakeOffscreenRenderer:
    def __init__(self, *a, **kw): pass
    def delete(self): pass
    def render(self, *a, **kw): return None, None


class _FakePyrender(ModuleType):
    def __init__(self):
        super().__init__("pyrender")
        self.OffscreenRenderer = _FakeOffscreenRenderer
        self.Mesh = type("Mesh", (object,), {"from_trimesh": staticmethod(lambda *a, **kw: None)})
        self.Scene = type("Scene", (object,), {"add": lambda *a, **kw: None, "add_node": lambda *a, **kw: None})
        self.Node = lambda *a, **kw: None
        self.DirectionalLight = lambda *a, **kw: None
        self.SpotLight = lambda *a, **kw: None
        self.PointLight = lambda *a, **kw: None
        self.IntrinsicsCamera = lambda *a, **kw: None
        self.PerspectiveCamera = lambda *a, **kw: None
        self.OrthographicCamera = lambda *a, **kw: None
        self.MetallicRoughnessMaterial = lambda *a, **kw: None
        self.RenderFlags = type("RenderFlags", (object,), {"RGBA": 1})


sys.modules["pyrender"] = _FakePyrender()


# --- 2. neutralize the overlay render (pure pyrender -> would crash each frame) ---
from wilor.utils.renderer import Renderer


def _fake_render_rgba_multiple(self, all_verts, cam_t=None, render_res=None,
                               is_right=None, **kw):
    rr = render_res
    try:
        rr = [int(x) for x in (rr.tolist() if hasattr(rr, "tolist") else rr)]
        W, H = rr[0], rr[1]
    except Exception:
        W, H = 256, 256
    # transparent RGBA -> the demo's overlay blend yields the input image, no crash
    return np.zeros((H, W, 4), dtype=np.float32)


Renderer.render_rgba_multiple = _fake_render_rgba_multiple


# --- 3. run the unmodified demo.py with the original argparse args ---
import runpy

runpy.run_path("demo.py", run_name="__main__")
