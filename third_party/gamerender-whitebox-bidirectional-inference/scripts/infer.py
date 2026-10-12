"""白模(white-box)微调 · 双向注意力(original_bidirectional) Wan2.2-Animate-2-14B LoRA 推理脚本。

本脚本是从 GameRender 项目 `scripts/train/validate_lora_animate2.py` 整理而来，专用于
"白模驱动视频 -> 渲染风格视频" 的生成任务（非流式，整段推理，底座为原始双向 Self-Attention）。

目录结构假设：
    <package_root>/
        diffsynth-studio/              依赖的本地 diffsynth 库（本包自带）
        checkpoints/animate2_base_scene00_lora/   白模微调 LoRA 权重（本包自带）
        demo_data/                     内置的 2 条真实验证样本，用于快速跑通（本包自带）
        scripts/infer.py               本脚本

注意：本包不包含底座大模型权重（Wan2.2-Animate-2-14B，约47GB），需要你自行准备并通过
--model-dir 指定其路径，目录结构需与原始一致：
    <model_dir>/wan_animate_2/wan_animate_2_bf16.safetensors
    <model_dir>/videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth
    <model_dir>/videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    <model_dir>/videomodel/Wan-AI/Wan2.1_VAE.pth
    <model_dir>/videomodel/Wan-AI/umt5-xxl/        (分词器目录)

用法1：跑内置 demo（验证环境是否搭好）：
    cd <package_root>
    CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
        --model-dir /path/to/Wan2.2-Animate-2-14B \
        --lora checkpoints/animate2_base_scene00_lora/step-7000.safetensors \
        --out-dir outputs/demo

用法2：自定义白模视频 + 外观参考图（真实使用场景）：
    CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
        --model-dir /path/to/Wan2.2-Animate-2-14B \
        --lora checkpoints/animate2_base_scene00_lora/step-7000.safetensors \
        --out-dir outputs/custom \
        --whitebox /path/to/your_whitebox.mp4 \
        --ref-image /path/to/ref_appearance.jpg \
        --prompt "人物外观描述：... 背景描述：..." \
        --prompt-ref "白盒渲染的游戏画面，画面中的角色在移动，相机视角持续变化。" \
        --num-inference-steps 40 --cfg-scale 3.0 --seed 0

每个样本输出: {样本名}_gen.mp4（生成）、{样本名}_gt.mp4（GT 渲染，若有）、{样本名}_whitebox.mp4（输入白模）。
"""
import argparse, json, os, shutil
import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(PACKAGE_ROOT, "demo_data")

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=False, default=None,
                        help="LoRA checkpoint (.safetensors)；传 none 或不传则用原始底座权重")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="底座大模型(Wan2.2-Animate-2-14B)目录路径，本包不自带，需自行准备，"
                             "目录结构需包含 wan_animate_2/ 和 videomodel/Wan-AI/ 子目录")
    parser.add_argument("--data-dir", default=DEFAULT_DATA,
                        help="已处理场景目录，包含 metadata_val.jsonl（默认使用包内 demo_data）")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--sample-ids", type=str, default=None,
                        help="逗号分隔的样本序号（在均匀取样结果中的下标），如 '0,1'；默认全部")
    parser.add_argument("--whitebox", type=str, default=None, help="自定义白模视频路径（驱动条件）")
    parser.add_argument("--ref-image", type=str, default=None, help="自定义外观参考图路径（渲染首帧等）")
    parser.add_argument("--prompt", type=str, default=None, help="自定义外观/背景描述")
    parser.add_argument("--prompt-ref", type=str, default=None, help="自定义驱动视频描述")
    parser.add_argument("--name", type=str, default="custom", help="输出文件名前缀")
    parser.add_argument("--gt", type=str, default=None, help="可选：GT 渲染视频路径，用于对比")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.whitebox is not None:
        assert args.ref_image is not None, "--whitebox 需要 --ref-image"
        samples = [{
            "video": args.gt,
            "animate2_reference_video": args.whitebox,
            "animate2_reference_image": args.ref_image,
            "prompt": args.prompt if args.prompt is not None else " ",
            "animate2_prompt_ref": args.prompt_ref if args.prompt_ref is not None else "白盒渲染的游戏画面，画面中的角色在移动，相机视角持续变化。",
            "_name": args.name,
        }]
        print(f"自定义输入: whitebox={args.whitebox} ref={args.ref_image}")
    else:
        with open(os.path.join(args.data_dir, "metadata_val.jsonl")) as f:
            metadata = [json.loads(l) for l in f if l.strip()]
        n = len(metadata)
        idxs = sorted(set(int(i * (n - 1) / max(1, args.num_samples - 1)) for i in range(args.num_samples)))
        if args.sample_ids is not None:
            sel = [int(x) for x in args.sample_ids.split(",")]
            idxs = [idxs[i] for i in sel]
        samples = [metadata[i] for i in idxs]
        print(f"val 共 {n} 条，选取 {idxs}")

    vram_config = {
        "offload_dtype": torch.bfloat16, "offload_device": "cpu",
        "onload_dtype": torch.bfloat16, "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(path=os.path.join(args.model_dir, "wan_animate_2/wan_animate_2_bf16.safetensors"), **vram_config),
            ModelConfig(path=os.path.join(args.model_dir, "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
            ModelConfig(path=os.path.join(args.model_dir, "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"), **vram_config),
            ModelConfig(path=os.path.join(args.model_dir, "videomodel/Wan-AI/Wan2.1_VAE.pth"), **vram_config),
        ],
        tokenizer_config=ModelConfig(os.path.join(args.model_dir, "videomodel/Wan-AI/umt5-xxl")),
    )
    if args.lora and args.lora.lower() != "none":
        pipe.load_lora(pipe.dit, args.lora, alpha=1)
    else:
        print("未加载 LoRA，使用原始底座权重")

    for sample_id, item in enumerate(samples):
        name = item.get("_name") or os.path.splitext(os.path.basename(item["video"]))[0].replace("_render", "")
        gt_path = os.path.join(args.data_dir, item["video"]) if item["video"] and not os.path.isabs(item["video"]) else item["video"]
        wb_path = item["animate2_reference_video"] if os.path.isabs(item["animate2_reference_video"]) else os.path.join(args.data_dir, item["animate2_reference_video"])
        ref_path = item["animate2_reference_image"] if os.path.isabs(item["animate2_reference_image"]) else os.path.join(args.data_dir, item["animate2_reference_image"])
        ref_image = Image.open(ref_path).convert("RGB")
        whitebox = VideoData(wb_path).raw_data()

        print(f"[{sample_id+1}/{len(samples)}] {name}: steps={args.num_inference_steps} cfg={args.cfg_scale}")
        video = pipe(
            prompt=item["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            animate2_prompt_ref=item["animate2_prompt_ref"],
            animate2_reference_image=ref_image,
            animate2_reference_video=whitebox[: args.num_frames],
            animate2_offload_kv=True,
            num_frames=args.num_frames, height=args.height, width=args.width,
            num_inference_steps=args.num_inference_steps, cfg_scale=args.cfg_scale,
            seed=args.seed, tiled=True,
        )
        save_video(video, os.path.join(args.out_dir, f"{name}_gen.mp4"), fps=15, quality=5)
        if gt_path and not os.path.exists(os.path.join(args.out_dir, f"{name}_gt.mp4")):
            shutil.copy(gt_path, os.path.join(args.out_dir, f"{name}_gt.mp4"))
        if not os.path.exists(os.path.join(args.out_dir, f"{name}_whitebox.mp4")):
            shutil.copy(wb_path, os.path.join(args.out_dir, f"{name}_whitebox.mp4"))
        print(f"  -> {args.out_dir}/{name}_gen.mp4")


if __name__ == "__main__":
    main()
