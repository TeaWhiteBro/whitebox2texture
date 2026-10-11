"""Split the generated video into frames and cut a masked reference crop of the object for Hunyuan3D-Paint.

Usage: [OBJECT=crate] python make_ref.py SCENE_DIR VIDEO.mp4 OUT_DIR REF_FRAME
Writes OUT_DIR/frames/####.png, OUT_DIR/ref_rgba.png (square, alpha = dilated white-box mask, or the video silhouette
from align_object.py if OUT_DIR/silhouette exists), OUT_DIR/ref.json.
REF_FRAME=auto picks the frame whose camera is closest to the object's front (theta = 0) without the object touching
the frame border; REF_FRAME=largest picks the frame where the object is biggest, unclipped and clear of the character
(SCENE_DIR/mask_character), which suits props seen from the side. Theta is measured around the object's own centre (bbox of SCENE_DIR/<object>.obj).
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from meshio_utils import mask_dir, object_name, read_obj  # noqa: E402


def camera_theta(c2w, centre):
    """Azimuth of the camera around the object: 0 = in front (-Y), +90 = +X. Same as Hunyuan azim."""
    x, y = c2w[0][3] - centre[0], c2w[1][3] - centre[1]
    return math.degrees(math.atan2(x, -y))


def main() -> None:
    scene, video, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    cams = json.loads((scene / "cameras.json").read_text())
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != cams["width"] or frame.shape[0] != cams["height"]:
            frame = cv2.resize(frame, (cams["width"], cams["height"]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "frames" / f"{count:04d}.png"), frame)
        count += 1
    V = read_obj(scene / f"{object_name()}.obj")[0]
    centre = (V.min(0) + V.max(0)) / 2
    thetas = [camera_theta(f["c2w_blender"], centre) for f in cams["frames"]]
    masks = [cv2.imread(str(mask_dir(scene) / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for i in range(count)]
    border = [m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() for m in masks]
    if sys.argv[4] == "auto":
        ref = int(np.argmin(np.abs(thetas) + 1e3 * np.array(border)))
    elif sys.argv[4] == "largest":
        score = []
        for i, m in enumerate(masks):
            char_path = scene / "mask_character" / f"{i:04d}.png"
            near = cv2.dilate(m.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
            clash = char_path.exists() and (cv2.imread(str(char_path), cv2.IMREAD_GRAYSCALE)[near] > 127).any()
            score.append(-1 if border[i] or clash else int(m.sum()))
        ref = int(np.argmax(score))
    else:
        ref = int(sys.argv[4])
    image = cv2.imread(str(out / "frames" / f"{ref:04d}.png"))
    silhouette = out / "silhouette" / f"{ref:04d}.png"
    if silhouette.exists():  # the video's own outline (align_object.py), when it drifts from the white box
        mask = cv2.imread(str(silhouette), cv2.IMREAD_GRAYSCALE) > 127
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    else:
        mask = cv2.dilate(masks[ref].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0  # stylised eaves overhang a little
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.56)
    rgba = np.zeros((2 * half, 2 * half, 4), np.uint8)
    x0, y0 = int(cx) - half, int(cy) - half
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + 2 * half, image.shape[1]), min(y0 + 2 * half, image.shape[0])
    rgba[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, :3] = image[sy0:sy1, sx0:sx1]
    rgba[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, 3] = mask[sy0:sy1, sx0:sx1] * 255
    rgba = cv2.resize(rgba, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out / "ref_rgba.png"), rgba)  # BGRA, as cv2 expects
    info = {"object": object_name(), "ref_frame": ref, "theta_deg": thetas[ref], "yaw_deg": -thetas[ref], "frames": count,
            "alpha": "silhouette" if silhouette.exists() else "white-box mask"}
    (out / "ref.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
