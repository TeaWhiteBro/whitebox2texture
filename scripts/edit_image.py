"""Generic first-frame edit with Qwen-Image-Edit-2511.

Usage: python edit_image.py SRC_IMAGE PROMPT_FILE OUT_DIR SEED [SEED ...]
Writes OUT_DIR/candidate_seed{S}.png for each seed (Qwen's own resolution, 1344x768 for an 832x480 input).
"""

import os
import sys
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

MODEL = os.environ.get("QWEN_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")  # local dir or HF repo id


def main() -> None:
    src, prompt_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    seeds = [int(s) for s in sys.argv[4:]]
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    if os.environ.get("IMGEDIT_OFFLOAD"):  # keep GPU use low when the box is busy
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    image = Image.open(src).convert("RGB")
    for seed in seeds:
        with torch.inference_mode():
            out = pipe(
                image=[image],
                prompt=prompt,
                negative_prompt=" ",
                true_cfg_scale=4.0,
                num_inference_steps=40,
                generator=torch.manual_seed(seed),
            ).images[0]
        path = out_dir / f"candidate_seed{seed}.png"
        out.save(path)
        print(path, out.size, flush=True)


if __name__ == "__main__":
    main()
