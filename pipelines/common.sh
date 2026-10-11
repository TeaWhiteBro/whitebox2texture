# Helpers shared by the pipelines (sourced after config.sh; expects GPUS=(...) and STAGES).

has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

# per_gpu FUNC ITEM...: run "FUNC ITEM GPU" for every item. Items are dealt round-robin over GPUS; different GPUs run
# in parallel, items that share a GPU run one after another. Fails if any item failed.
per_gpu() {
  local fn=$1 i=0 g it p fail=0 pids=()
  shift
  declare -A queue=()
  for it in "$@"; do g=${GPUS[$((i % ${#GPUS[@]}))]}; queue[$g]+="$it "; i=$((i + 1)); done
  for g in "${!queue[@]}"; do (for it in ${queue[$g]}; do "$fn" "$it" "$g"; done) & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  return $fail
}

# to_ref CANDIDATE OUT: Qwen output (1344x768) -> the 832x480 first frame, exactly as for the published runs.
to_ref() {
  "$PY_HY3D" -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').resize((832, 480), Image.Resampling.LANCZOS).save(sys.argv[2])" "$1" "$2"
}
