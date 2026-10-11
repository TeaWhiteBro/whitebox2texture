"""Collect eval_l1.json files into one Markdown table (masked L1 vs the world-model video, 0-255, lower is better).

Usage: python summarize_l1.py RUN_DIR [RUN_DIR ...]   (each RUN_DIR holds tex/ or tex_<object>/ folders)
"""

import json
import sys
from pathlib import Path


def main() -> None:
    print("| run | object | Hunyuan only (lit) | + video projection (unlit) | aligned: Hunyuan / + projection | coverage |")
    print("|---|---|---|---|---|---|")
    for run in map(Path, sys.argv[1:]):
        for tex in sorted(run.glob("tex*")):
            if not (tex / "eval_l1.json").exists():
                continue
            scores = {Path(k).name: v for k, v in json.loads((tex / "eval_l1.json").read_text()).items()}
            hy, fu = scores["hunyuan_lit_original"], scores["fused_unlit_original"]
            cover = json.loads((tex / "proj" / "stats.json").read_text())["fraction_projected"]
            aligned = f"{hy['aligned']:.1f} / {fu['aligned']:.1f}" if hy["aligned"] is not None else "-"
            obj = tex.name[4:] or "house"
            print(f"| {run.name} | {obj} | {hy['raw']:.1f} | {fu['raw']:.1f} | {aligned} | {cover:.0%} |")


if __name__ == "__main__":
    main()
