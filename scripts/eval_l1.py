"""Masked L1 (0-255) between re-rendered textured object and the generated video, per variant, over all frames.

Usage: [OBJECT=crate] python eval_l1.py SCENE_DIR TEX_DIR VARIANT [VARIANT ...]   (VARIANT = folder under TEX_DIR/render)
Compares inside the eroded white-box mask. If TEX_DIR/align.json exists, also reports an aligned score: the render is
warped into the video's frame with the per-frame affine and compared inside warped mask & video silhouette, which
separates texture errors from the video model misplacing the object.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from meshio_utils import mask_dir  # noqa: E402


def main() -> None:
    scene, tex = Path(sys.argv[1]), Path(sys.argv[2])
    align_path = tex / "align.json"
    align = json.loads(align_path.read_text()) if align_path.exists() else {}
    frames = sorted((tex / "frames").glob("*.png"))
    result = {}
    for variant in sys.argv[3:]:
        raw, aligned = [], []
        for p in frames:
            f = int(p.stem)
            video = cv2.imread(str(p)).astype(np.float32)
            render = cv2.imread(str(tex / "render" / variant / p.name)).astype(np.float32)
            mask = cv2.imread(str(mask_dir(scene) / p.name), cv2.IMREAD_GRAYSCALE) > 127
            mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
            if mask.sum() < 200:  # object out of view in this frame
                continue
            raw.append(np.abs(render - video)[mask > 0].mean())
            if align:
                A = np.array(align[str(f)]["affine"], np.float32)
                size = video.shape[1::-1]
                warped = cv2.warpAffine(render, A, size, flags=cv2.INTER_LINEAR)
                m = cv2.warpAffine(mask, A, size, flags=cv2.INTER_NEAREST) > 0
                m &= cv2.imread(str(tex / "silhouette" / p.name), cv2.IMREAD_GRAYSCALE) > 127
                if m.sum() >= 200:
                    aligned.append(np.abs(warped - video)[m].mean())
        result[variant] = {"raw": float(np.mean(raw)), "aligned": float(np.mean(aligned)) if aligned else None}
        print(Path(variant).name, {k: None if v is None else round(v, 1) for k, v in result[variant].items()}, flush=True)
    (tex / "eval_l1.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
