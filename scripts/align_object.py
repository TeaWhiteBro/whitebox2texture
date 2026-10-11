"""Register the generated video to the white-box geometry for one object, frame by frame.

The world model does not always put small objects exactly where the white box is (the crate drifts ~20 px). For each
frame this segments the object in the video with GrabCut, seeded by the white-box mask, then fits a 2D affine warp
(white-box pixel -> video pixel) with ECC on the two silhouettes. Where the object leaves the frame, the white-box
silhouette is extended past the border by projecting the mesh, and ECC ignores video pixels outside the frame, so a
cut-off object does not squash the fit. A fit is kept only if it clearly improves the overlap and is plausible (scale
0.8-1.25, shift under a third of the object's size); otherwise the warp is interpolated from good frames at most 6
frames away, or left as identity.

Usage: OBJECT=crate python align_object.py SCENE_DIR VIDEO.mp4 OUT_DIR
Writes OUT_DIR/silhouette/####.png and OUT_DIR/align.json ({frame: 2x3 affine}, IoU before/after).
make_ref.py and project_frames.py pick these up automatically when they exist in the same OUT_DIR.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from meshio_utils import mask_dir, object_name, read_obj  # noqa: E402

PAD = 240


def segment(frame, prior):
    """GrabCut in a window around the white-box mask; returns the video silhouette (bool, full frame)."""
    h, w = prior.shape
    ys, xs = np.nonzero(prior)
    size = max(xs.max() - xs.min(), ys.max() - ys.min())
    pad = int(0.5 * size) + 20
    x0, x1, y0, y1 = max(xs.min() - pad, 0), min(xs.max() + pad, w), max(ys.min() - pad, 0), min(ys.max() + pad, h)
    roi, p = frame[y0:y1, x0:x1], prior[y0:y1, x0:x1].astype(np.uint8)
    reach = int(0.3 * size) + 6  # how far the generated object may sit from the white box
    gc = np.full(p.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(p, np.ones((2 * reach + 1,) * 2, np.uint8)) > 0] = cv2.GC_PR_BGD
    gc[p > 0] = cv2.GC_PR_FGD
    core = int(0.3 * size)
    gc[cv2.erode(p, np.ones((2 * core + 1,) * 2, np.uint8)) > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(roi, gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)  # keep the blob that overlaps the prior most
    if n > 1:
        overlap = [(labels == k)[p > 0].sum() for k in range(1, n)]
        fg = (labels == 1 + int(np.argmax(overlap))).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out = np.zeros((h, w), bool)
    out[y0:y1, x0:x1] = fg > 0
    return out


def project_silhouette(V, F, cam, c2w, shape):
    """Union of the mesh's projected triangles on a canvas padded by PAD on every side (no occlusion)."""
    w2c = np.linalg.inv(np.array(c2w))
    Pc = V @ w2c[:3, :3].T + w2c[:3, 3]
    z = np.maximum(-Pc[:, 2], 1e-3)
    uv = np.stack([cam["fx"] * Pc[:, 0] / z + cam["cx"], cam["cy"] - cam["fy"] * Pc[:, 1] / z], 1) + PAD
    canvas = np.zeros((shape[0] + 2 * PAD, shape[1] + 2 * PAD), np.uint8)
    cv2.fillPoly(canvas, list(np.round(uv[F] * 4).astype(np.int32)), 1, shift=2)
    return canvas > 0


def fit_affine(prior_padded, sil):
    """Affine A (2x3, frame pixels) with sil(A [x,y,1]) ~ prior(x,y), i.e. white-box pixel -> video pixel."""
    h, w = sil.shape
    src = prior_padded.astype(np.float32)
    dst = np.zeros_like(src)
    dst[PAD:PAD + h, PAD:PAD + w] = sil
    ys, xs = np.nonzero(prior_padded[PAD:PAD + h, PAD:PAD + w])
    yd, xd = np.nonzero(sil)
    sigma = max(2.0, 0.04 * max(xs.ptp(), ys.ptp()))
    inside = np.zeros(dst.shape, np.uint8)  # video pixels we actually have, minus the blurred border band
    band = int(3 * sigma) + 1
    inside[PAD + band:PAD + h - band, PAD + band:PAD + w - band] = 1
    A = np.array([[1, 0, xd.mean() - xs.mean()], [0, 1, yd.mean() - ys.mean()]], np.float32)
    a = cv2.GaussianBlur(src, (0, 0), sigma)
    b = cv2.GaussianBlur(dst, (0, 0), sigma)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    try:  # canvas coordinates differ from frame coordinates by PAD on both sides, which an affine absorbs
        Ac = A.copy()
        Ac[:, 2] += PAD - A[:, :2] @ np.array([PAD, PAD], np.float32)
        _, Ac = cv2.findTransformECC(a, b, Ac, cv2.MOTION_AFFINE, criteria, inside, 1)
        A = Ac.copy()
        A[:, 2] = Ac[:, 2] + Ac[:, :2] @ np.array([PAD, PAD], np.float32) - PAD
    except cv2.error:
        pass  # keep the centroid translation
    return A


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> None:
    scene, video, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    (out / "silhouette").mkdir(parents=True, exist_ok=True)
    masks = mask_dir(scene)
    cams = json.loads((scene / "cameras.json").read_text())
    V, _, F, _, _ = read_obj(scene / f"{object_name()}.obj")
    priors, sils, affines = [], [], []
    cap = cv2.VideoCapture(video)
    while True:
        ok, image = cap.read()
        if not ok:
            break
        f = len(priors)
        prior = cv2.imread(str(masks / f"{f:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        if image.shape[:2] != prior.shape:
            image = cv2.resize(image, prior.shape[::-1], interpolation=cv2.INTER_AREA)
        visible = prior.sum() >= 200
        sil = segment(image, prior) if visible else np.zeros_like(prior)
        cv2.imwrite(str(out / "silhouette" / f"{f:04d}.png"), sil.astype(np.uint8) * 255)
        padded = project_silhouette(V, F, cams, cams["frames"][f]["c2w_blender"], prior.shape)
        padded[PAD:PAD + prior.shape[0], PAD:PAD + prior.shape[1]] = prior  # rendered mask keeps occlusion in frame
        priors.append(prior)
        sils.append(sil)
        affines.append(fit_affine(padded, sil) if visible else np.eye(3, dtype=np.float32)[:2])

    def warp(A, prior):
        return cv2.warpAffine(prior.astype(np.uint8), A, prior.shape[::-1], flags=cv2.INTER_NEAREST) > 0

    before = [iou(p, s) for p, s in zip(priors, sils)]
    after = [iou(warp(A, p), s) for A, p, s in zip(affines, priors, sils)]
    def plausible(A, prior):
        ys, xs = np.nonzero(prior)
        if len(xs) < 200:
            return False
        scale = np.sqrt(abs(np.linalg.det(A[:, :2])))
        centre = np.array([xs.mean(), ys.mean()])
        shift = np.linalg.norm(A[:, :2] @ centre + A[:, 2] - centre)
        return 0.8 <= scale <= 1.25 and shift <= max(np.ptp(xs), np.ptp(ys)) / 3

    good = [a >= max(b + 0.02, 0.75) and plausible(A, p) for a, b, A, p in zip(after, before, affines, priors)]
    idx = np.flatnonzero(good)
    stack = np.array(affines)
    result = {}
    for f in range(len(priors)):
        A = stack[f]
        if not good[f]:  # the warp changes slowly over time: borrow it from nearby good frames, else trust the white box
            lo, hi = idx[idx < f], idx[idx > f]
            if len(lo) and len(hi) and f - lo[-1] <= 6 and hi[0] - f <= 6:
                A = np.stack([np.interp(f, idx, stack[idx, i, j]) for i in range(2) for j in range(3)]).reshape(2, 3)
            else:
                A = np.eye(3)[:2]
        ys, xs = np.nonzero(priors[f])
        shift = (A @ np.array([xs.mean(), ys.mean(), 1.0]) - [xs.mean(), ys.mean()]).tolist() if len(xs) >= 200 else None
        result[f] = {"affine": A.tolist(), "iou_before": before[f], "iou_after": iou(warp(A, priors[f]), sils[f]),
                     "interpolated": bool(not good[f]), "centre_shift_px": shift}
        print(f, {k: v if isinstance(v, bool) or v is None else np.round(v, 3).tolist() for k, v in result[f].items() if k != "affine"})
    (out / "align.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
