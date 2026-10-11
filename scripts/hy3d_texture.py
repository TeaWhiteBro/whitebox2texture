"""Hunyuan3D-Paint 2.1 on a white-box mesh, with one masked video frame as the appearance reference.

Usage (hy3d env, HY3D_REPO set): python hy3d_texture.py MESH_OBJ REF_RGBA YAW_DEG OUT_DIR [--views 6] [--res 512]
MESH_OBJ is in Blender world coordinates. YAW_DEG rotates it about Z so the reference view becomes Hunyuan's
front camera (azim 0). Writes OUT_DIR/textured_world.obj (+ .mtl and PBR PNGs) back in Blender world coordinates.
"""

import argparse
import os
import sys
import types
from pathlib import Path

HY = os.environ["HY3D_REPO"]  # Hunyuan3D-2.1 checkout (config.sh)
sys.modules.setdefault("bpy", types.ModuleType("bpy"))  # only used by Hunyuan's glb export; Blender does that for us
sys.path[:0] = [f"{HY}/hy3dpaint", HY, str(Path(__file__).parent)]

from torchvision_fix import apply_fix  # noqa: E402  (basicsr needs torchvision.transforms.functional_tensor)

apply_fix()

import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

from meshio_utils import blender_to_hunyuan, hunyuan_to_blender, read_obj, write_obj  # noqa: E402
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("ref")
    parser.add_argument("yaw", type=float)
    parser.add_argument("out")
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--res", type=int, default=512)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    V, _, F, _, _ = read_obj(args.mesh)
    welded = trimesh.Trimesh(blender_to_hunyuan(V, args.yaw), F, process=True)  # weld so xatlas makes large charts
    welded.export(out / "input_hunyuan.obj")

    config = Hunyuan3DPaintConfig(max_num_view=args.views, resolution=args.res)
    config.multiview_cfg_path = f"{HY}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    config.realesrgan_ckpt_path = f"{HY}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    pipeline = Hunyuan3DPaintPipeline(config)
    pipeline(
        mesh_path=str(out / "input_hunyuan.obj"),
        image_path=Image.open(args.ref),
        output_mesh_path=str(out / "textured_hunyuan.obj"),
        use_remesh=False,
        save_glb=False,
    )
    Vh, VT, Fh, FT, header = read_obj(out / "textured_hunyuan.obj")
    write_obj(out / "textured_world.obj", hunyuan_to_blender(Vh, args.yaw), Fh, VT, FT, header)
    print("wrote", out / "textured_world.obj", "verts", len(Vh), "faces", len(Fh), os.listdir(out))


if __name__ == "__main__":
    main()
