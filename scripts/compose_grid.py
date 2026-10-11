"""Tile several frame folders / MP4s into one labelled grid video.

Usage: python compose_grid.py OUT.mp4 COLS FPS SRC:LABEL [SRC:LABEL ...]
SRC is a folder of ####.png frames or an .mp4. Frames are taken in lockstep; the shortest source sets the length.
GRID_TILE=WxH (default 832x480) sets the tile size.
"""

import os
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def frames_of(src):
    path = Path(src)
    if path.is_dir():
        for p in sorted(path.glob("*.png")):
            yield Image.open(p).convert("RGB")
    else:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_image()


def label(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    draw.rectangle((0, 0, draw.textlength(text, font=font) + 16, 34), fill=(0, 0, 0))
    draw.text((8, 4), text, fill=(255, 255, 255), font=font)
    return image


def main() -> None:
    out, cols, fps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    items = [arg.split(":", 1) for arg in sys.argv[4:]]
    streams = [frames_of(src) for src, _ in items]
    tile = tuple(int(v) for v in os.environ.get("GRID_TILE", "832x480").split("x"))
    rows = (len(items) + cols - 1) // cols
    with av.open(out, mode="w") as dst:
        stream = dst.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = tile[0] * cols, tile[1] * rows, "yuv420p"
        stream.options = {"crf": "18"}
        count = 0
        while True:
            frames = [next(s, None) for s in streams]
            if any(f is None for f in frames):
                break
            canvas = Image.new("RGB", (stream.width, stream.height))
            for k, (frame, (_, text)) in enumerate(zip(frames, items)):
                canvas.paste(label(frame.resize(tile, Image.Resampling.LANCZOS), text), ((k % cols) * tile[0], (k // cols) * tile[1]))
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                dst.mux(packet)
            count += 1
        for packet in stream.encode():
            dst.mux(packet)
    print(out, count, "frames", stream.width, "x", stream.height)


if __name__ == "__main__":
    main()
