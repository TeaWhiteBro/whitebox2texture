"""Encode a folder of PNG frames into an H.264 MP4. Usage: python encode_video.py FRAME_DIR OUT.mp4 [FPS]"""

import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image


def main() -> None:
    frames = sorted(Path(sys.argv[1]).glob("*.png"))
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    first = Image.open(frames[0])
    with av.open(sys.argv[2], mode="w") as dst:
        stream = dst.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = first.width, first.height, "yuv420p"
        stream.options = {"crf": "16"}
        for path in frames:
            image = np.asarray(Image.open(path).convert("RGB"))
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    print(sys.argv[2], len(frames), "frames")


if __name__ == "__main__":
    main()
