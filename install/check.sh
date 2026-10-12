#!/bin/bash
# Check that every component the pipelines need is in place (no GPU work, no downloads).
source "$(dirname "$(realpath "$0")")/../config.sh"
ok=0
check() { if eval "$2" > /dev/null 2>&1; then echo "  ok    $1"; else echo "  MISS  $1"; ok=1; fi; }
optional() { if eval "$2" > /dev/null 2>&1; then echo "  ok    $1"; else echo "  --    $1: not installed"; fi; }
echo "config: W2T_HOME=$W2T_HOME RUNS=$RUNS CYCLES_DEVICE=$CYCLES_DEVICE"
check "Blender 4.5 ($BLENDER)" "blender -b --factory-startup --python-expr 'import bpy; assert bpy.app.version[:2] == (4, 5)'"
check "hy3d env ($PY_HY3D)" "'$PY_HY3D' -c 'import torch, av, cv2, trimesh, xatlas'"
check "Hunyuan3D-2.1 + extensions ($HY3D_REPO)" "cd '$HY3D_REPO/hy3dpaint' && '$PY_HY3D' -c 'import torch, custom_rasterizer, DifferentiableRenderer.mesh_inpaint_processor'"
check "RealESRGAN weights" "test -s '$HY3D_REPO/hy3dpaint/ckpt/RealESRGAN_x4plus.pth'"
check "Hunyuan3D-Paint + DINOv2 weights ($HF_HOME)" "HF_HUB_OFFLINE=1 '$PY_HY3D' -c 'import huggingface_hub as h; h.snapshot_download(\"tencent/Hunyuan3D-2.1\", allow_patterns=[\"hunyuan3d-paintpbr-v2-1/*\"]); h.snapshot_download(\"facebook/dinov2-giant\")'"
check "GameRender package + LoRA ($GAMERENDER_REPO)" "test -f '$GAMERENDER_REPO/scripts/infer.py' -a -f '$WAN_LORA'"
check "wananimate env ($PY_WAN)" "'$PY_WAN' -c 'from diffsynth.pipelines.wan_video import WanVideoPipeline'"
check "Wan2.2-Animate-2-14B weights ($WAN_MODEL_DIR)" "test -s '$WAN_MODEL_DIR/wan_animate_2/wan_animate_2_bf16.safetensors' -a -s '$WAN_MODEL_DIR/videomodel/Wan-AI/Wan2.1_VAE.pth' -a -s '$WAN_MODEL_DIR/videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth'"
echo "optional (install/04_imgedit.sh, only to regenerate first frames with REF=seed):"
optional "imgedit env ($PY_IMGEDIT)" "'$PY_IMGEDIT' -c 'from diffusers import QwenImageEditPlusPipeline'"
optional "Qwen-Image-Edit-2511 weights ($QWEN_EDIT_MODEL)" "test -s '$QWEN_EDIT_MODEL/model_index.json'"
[ $ok = 0 ] && echo "all required components ok" || echo "some required components are missing (MISS above)"
exit $ok
