"""Keep an image edit only inside one object's region, so a first-frame variant differs from the original only there.

Usage: python paste_region.py ORIGINAL.png EDITED.png MASK.png OUT.png [DILATE_PX]
EDITED is resized to ORIGINAL's size; MASK (white-box mask of that frame) is dilated by DILATE_PX (default 14, the
stylised object overhangs its white box), grown into nearby pixels the edit changed (a stylised chimney or thatch can
reach well past the white box), feathered, and only that region is taken from EDITED.
"""

import sys

import cv2
import numpy as np


def main() -> None:
    original = cv2.imread(sys.argv[1]).astype(np.float32)
    edited = cv2.imread(sys.argv[2])
    edited = cv2.resize(edited, original.shape[1::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = (cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    d = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    ys, xs = np.nonzero(mask)
    window = np.zeros_like(mask)
    window[max(ys.min() - 60, 0):ys.max() + 60, max(xs.min() - 60, 0):xs.max() + 60] = 1
    changed = (np.abs(edited - original).max(2) > 40).astype(np.uint8) & window
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels = cv2.connectedComponents(changed | mask)
    keep = np.unique(labels[mask > 0])
    mask = cv2.dilate(np.isin(labels, keep[keep > 0]).astype(np.uint8), np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)[..., None]
    cv2.imwrite(sys.argv[4], (alpha * edited + (1 - alpha) * original).round().astype(np.uint8))
    print(sys.argv[4], "changed pixels:", int((alpha[..., 0] > 0.01).sum()))


if __name__ == "__main__":
    main()
