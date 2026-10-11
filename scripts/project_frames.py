"""Project selected video frames into the UV atlas of a textured mesh and fuse them with its generated albedo.

Usage: [OBJECT=crate] python project_frames.py SCENE_DIR FRAMES_DIR MESH_OBJ BASE_ALBEDO OUT_DIR FRAME [FRAME ...]
MESH_OBJ must be in Blender world coordinates with UVs (e.g. textured_world.obj from hy3d_texture.py).
Visible texels take video colour (view-angle / distance weighted, depth-tested against the white-box depth,
inside the eroded object mask); unseen texels keep the generated albedo, colour-matched to the projected part.
If FRAMES_DIR/../align.json exists (align_object.py), video colours are fetched through its per-frame 2D warp.
If SCENE_DIR/mask_character exists, texels that land near the character are skipped: the generated hero's hair,
coat and weapons reach well past the white-box mannequin.
VIEW_SELECT=patch takes each planar patch (texels grouped by normal) from its single best frame instead of averaging
all frames; use it when frames disagree (small objects whose generated look drifts), which otherwise shows up as ghosting.
Writes OUT_DIR/{albedo_projected.png, albedo_fused.png, coverage.png, stats.json}.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from meshio_utils import mask_dir, read_obj  # noqa: E402

COS_POWER = 4.0
SELECT = os.environ.get("VIEW_SELECT", "blend")
CHAR_MARGIN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))


def rasterize_uv(V, VT, FV, FT, size):
    """Per-texel world position and face normal (flat), plus validity."""
    pos = np.zeros((size, size, 3), np.float32)
    nrm = np.zeros((size, size, 3), np.float32)
    valid = np.zeros((size, size), bool)
    face_n = np.cross(V[FV[:, 1]] - V[FV[:, 0]], V[FV[:, 2]] - V[FV[:, 0]])
    face_n /= np.linalg.norm(face_n, axis=1, keepdims=True) + 1e-12
    px = np.stack([VT[:, 0] * size - 0.5, (1.0 - VT[:, 1]) * size - 0.5], 1)
    for f in range(len(FV)):
        a, b, c = px[FT[f]]
        x0, x1 = int(np.floor(min(a[0], b[0], c[0]))), int(np.ceil(max(a[0], b[0], c[0])))
        y0, y1 = int(np.floor(min(a[1], b[1], c[1]))), int(np.ceil(max(a[1], b[1], c[1])))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        l0 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / det
        l1 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / det
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any():
            continue
        p = l0[..., None] * V[FV[f, 0]] + l1[..., None] * V[FV[f, 1]] + l2[..., None] * V[FV[f, 2]]
        ys, xs = gy[inside], gx[inside]
        pos[ys, xs] = p[inside]
        nrm[ys, xs] = face_n[f]
        valid[ys, xs] = True
    return pos, nrm, valid


def bilinear(image, x, y):
    x = np.clip(x, 0, image.shape[1] - 1.001)
    y = np.clip(y, 0, image.shape[0] - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def fill_gutters(image, valid, iterations=16):
    """Grow valid texels outwards so bilinear/mip sampling at chart borders does not pick up stale colours."""
    image, valid = image.copy(), valid.copy()
    kernel = np.ones((3, 3), np.float32)
    for _ in range(iterations):
        w = cv2.filter2D(valid.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        s = cv2.filter2D(image * valid[..., None], -1, kernel, borderType=cv2.BORDER_CONSTANT)
        grow = (~valid) & (w > 0)
        image[grow] = s[grow] / w[grow][:, None]
        valid |= grow
    return image


def main() -> None:
    scene, frames_dir, mesh_path, base_path, out = map(Path, sys.argv[1:6])
    frame_ids = [int(f) for f in sys.argv[6:]]
    out.mkdir(parents=True, exist_ok=True)
    cams = json.loads((scene / "cameras.json").read_text())
    align_path = frames_dir.parent / "align.json"
    align = json.loads(align_path.read_text()) if align_path.exists() else {}
    W, H = cams["width"], cams["height"]
    fx, fy, cx, cy = cams["fx"], cams["fy"], cams["cx"], cams["cy"]

    V, VT, FV, FT, _ = read_obj(mesh_path)
    base = cv2.cvtColor(cv2.imread(str(base_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    size = base.shape[0]
    pos, nrm, valid = rasterize_uv(V, VT, FV, FT, size)
    P = pos[valid]
    N = nrm[valid]

    acc = np.zeros((len(P), 3), np.float64)
    wsum = np.zeros(len(P), np.float64)
    best_cos = np.zeros(len(P), np.float64)
    per_frame = {}
    weights, colors = [], []
    erode = np.ones((5, 5), np.uint8)
    for f in frame_ids:
        c2w = np.array(cams["frames"][f]["c2w_blender"])
        w2c = np.linalg.inv(c2w)
        Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = -Pc[:, 2]
        u = fx * Pc[:, 0] / np.maximum(z, 1e-6) + cx
        v = cy - fy * Pc[:, 1] / np.maximum(z, 1e-6)
        inframe = (z > 0.1) & (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
        ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1)
        depth = cv2.imread(str(scene / "depth" / f"{f:04d}.exr"), cv2.IMREAD_UNCHANGED)
        depth = depth[..., 0] if depth.ndim == 3 else depth
        visible = np.abs(depth[vi, ui] - z) < 0.02 + 0.01 * z
        mask = cv2.imread(str(mask_dir(scene) / f"{f:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        mask = cv2.erode(mask.astype(np.uint8), erode) > 0  # keep away from silhouettes (video is not pixel exact)
        char_path = scene / "mask_character" / f"{f:04d}.png"
        if char_path.exists():
            char = cv2.imread(str(char_path), cv2.IMREAD_GRAYSCALE) > 127
            mask &= cv2.dilate(char.astype(np.uint8), CHAR_MARGIN) == 0
        view = c2w[:3, 3] - P
        cos = np.einsum("ij,ij->i", N, view) / np.linalg.norm(view, axis=1)
        ok = inframe & visible & mask[vi, ui] & (cos > 0.1)
        frame = cv2.cvtColor(cv2.imread(str(frames_dir / f"{f:04d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        A = np.array(align[str(f)]["affine"]) if str(f) in align else np.eye(3)[:2]  # white-box px -> video px
        color = bilinear(frame, A[0, 0] * u + A[0, 1] * v + A[0, 2] - 0.5, A[1, 0] * u + A[1, 1] * v + A[1, 2] - 0.5)
        w = np.where(ok, np.clip(cos, 0, 1) ** COS_POWER / np.maximum(z, 1e-3) ** 2, 0.0)
        if SELECT == "patch":
            weights.append(w.astype(np.float32))
            colors.append(color.astype(np.float32))
        acc += w[:, None] * color
        wsum += w
        best_cos = np.maximum(best_cos, np.where(ok, cos, 0))
        per_frame[f] = int(ok.sum())

    covered = wsum > 0
    proj = np.zeros_like(acc)
    proj[covered] = acc[covered] / wsum[covered, None]
    if SELECT == "patch":  # seams then fall on mesh edges instead of ghosting across a face
        _, patch = np.unique(np.round(N * 4).astype(int), axis=0, return_inverse=True)
        patch = patch.reshape(-1)
        quality = [align.get(str(f), {}).get("iou_after", 1.0) ** 8 for f in frame_ids]  # distrust poorly aligned frames
        score = np.stack([np.bincount(patch, w, patch.max() + 1) * q for w, q in zip(weights, quality)], 1)
        pick = score.argmax(1)[patch]
        rows = np.flatnonzero(np.stack(weights)[pick, np.arange(len(P))] > 0)  # the rest keeps the blend
        proj[rows] = np.stack(colors)[pick[rows], rows]
        per_patch = {int(k): frame_ids[int(i)] for k, i in enumerate(score.argmax(1)) if score[k].max() > 0}
    alpha = np.clip((best_cos - 0.1) / 0.3, 0, 1) * covered  # fade out grazing-angle texels

    base_v = base[valid].astype(np.float64)
    sure = alpha > 0.9
    mu_p, sd_p = proj[sure].mean(0), proj[sure].std(0) + 1e-6
    mu_b, sd_b = base_v[sure].mean(0), base_v[sure].std(0) + 1e-6
    base_matched = np.clip((base_v - mu_b) / sd_b * sd_p + mu_p, 0, 1)

    def to_image(values, fallback):
        img = fallback.copy()
        img[valid] = values
        return img

    alpha_img = np.zeros((size, size), np.float32)
    alpha_img[valid] = alpha
    alpha_img = cv2.GaussianBlur(alpha_img, (0, 0), 2.0) * valid  # soften the seam between projected and generated
    proj_img = to_image(proj, base)
    matched_img = to_image(base_matched, base)
    fused = alpha_img[..., None] * proj_img + (1 - alpha_img[..., None]) * matched_img
    fused = fill_gutters(fused.astype(np.float32), valid)
    covered_img = np.zeros((size, size), bool)
    covered_img[valid] = covered
    proj_vis = fill_gutters(np.where(covered_img[..., None], proj_img, 0.0).astype(np.float32), valid)

    def save(name, img):
        cv2.imwrite(str(out / name), cv2.cvtColor((np.clip(img, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))

    save("albedo_fused.png", fused)
    save("albedo_projected.png", proj_vis)
    save("coverage.png", np.repeat(alpha_img[..., None], 3, 2))
    stats = {
        "frames": frame_ids, "aligned": bool(align), "view_select": SELECT, "texels_valid": int(valid.sum()), "texels_projected": int(covered.sum()),
        "fraction_projected": float(covered.mean()), "fraction_confident": float(sure.mean()),
        "texels_per_frame": per_frame, "frame_per_patch": per_patch if SELECT == "patch" else None, "colour_match": {"proj_mean": mu_p.tolist(), "base_mean": mu_b.tolist()},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
