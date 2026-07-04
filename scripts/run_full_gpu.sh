#!/usr/bin/env bash
# =============================================================
# scripts/run_full_gpu.sh
# Chạy TOÀN BỘ thí nghiệm Giai đoạn 1 trên GPU (vast.ai).
#
# Đặc điểm
# --------
# 1. KIỂM TRA CUDA ngay đầu (tránh vô tình train trên CPU, tốn time).
# 2. Đọc config.yaml → build --scenarios, --cap-per-class, --epochs.
# 3. Gọi ``python -m src.run_experiments --auto-resume`` (đã có logic
#    skip_configs đã chạy + derive Phase A winner).
# 4. SAU KHI PYTHON EXIT (kể cả lỗi):
#    a) git add + git commit artifacts/ (nếu là git repo).
#    b) git push — nếu OK thì xong; nếu fail (no credential) → in
#       gợi ý lệnh rsync/scp để user tự backup thủ công.
# 5. KHÔNG tự ý xóa artifacts/ hay checkpoint.
#
# Chế độ chạy
# -----------
#   bash scripts/run_full_gpu.sh                   # mặc định: từ config.yaml
#   bash scripts/run_full_gpu.sh --epochs 30       # override số epoch
#   bash scripts/run_full_gpu.sh --no-git          # bỏ qua bước git backup
#
# YÊU CẦU
# --------
# • Đã cài requirements (torch CUDA, torch-geometric, scikit-learn…).
# • Đã tải data (xem scripts/download_all.sh).
# • Đã chạy scripts/eda_all_scenarios.py để chốt cap_per_class.
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -------- Args --------
CONFIG_PATH="config.yaml"
DO_GIT_BACKUP=1
OVERRIDE_EPOCHS=""
OVERRIDE_CAP=""

usage() {
  cat <<EOF
Cách dùng:
  bash scripts/run_full_gpu.sh                   # mặc định
  bash scripts/run_full_gpu.sh --config PATH     # đổi config.yaml
  bash scripts/run_full_gpu.sh --epochs 30       # override số epoch (mặc định từ config)
  bash scripts/run_full_gpu.sh --cap 20000       # override cap_per_class (mặc định từ config)
  bash scripts/run_full_gpu.sh --no-git          # tắt git backup cuối run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)       CONFIG_PATH="$2"; shift ;;
    --epochs)       OVERRIDE_EPOCHS="$2"; shift ;;
    --cap)          OVERRIDE_CAP="$2"; shift ;;
    --no-git)       DO_GIT_BACKUP=0 ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "[ERR] Tham số lạ: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

# -------- Tool check --------
command -v python >/dev/null 2>&1 || { echo "[ERR] Thiếu 'python'." >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "[ERR] Thiếu 'git'." >&2; exit 1; }
[[ -f "${CONFIG_PATH}" ]] || { echo "[ERR] Không thấy config: ${CONFIG_PATH}" >&2; exit 1; }

# -------- 1) Kiểm tra CUDA --------
echo "============================================================"
echo " STEP 1  ·  Kiểm tra CUDA + in thông tin GPU"
echo "============================================================"
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("[ERR] Chưa cài torch. Cài theo hướng dẫn trong requirements.txt.", file=sys.stderr)
    sys.exit(1)

print(f"  torch.__version__    : {torch.__version__}")
print(f"  torch.version.cuda   : {torch.version.cuda}")
print(f"  CUDA available       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"  #GPU                : {n}")
    for i in range(n):
        print(f"  GPU {i}              : {torch.cuda.get_device_name(i)}")
    print("  → CUDA OK — sẽ train trên GPU.")
else:
    print("  [FATAL] CUDA KHÔNG khả dụng — refuse to run on CPU.", file=sys.stderr)
    print("  Hãy đảm bảo instance vast.ai có GPU và torch+CUDA đã cài đúng.", file=sys.stderr)
    sys.exit(2)
PY
echo

# -------- 2) Build args từ config --------
echo "============================================================"
echo " STEP 2  ·  Đọc config + build CLI args"
echo "============================================================"

# Parse experiments.scenarios thành danh sách ``name=PATH`` (1 path / dòng).
SCEN_PAIRS="$(awk '
  /^experiments:/ { in_exp=1; next }
  in_exp && /^  scenarios:/ { in_sc=1; next }
  in_sc {
    if (/^    - name:/) {
      val = $0; sub(/^[[:space:]]*-[[:space:]]*name:[[:space:]]*/, "", val);
      if ((getline p_line) > 0) {
        p_val = p_line; sub(/^[[:space:]]*path:[[:space:]]*/, "", p_val);
        print val "=" p_val;
      }
      next;
    }
    if (/^  [a-z]/) { in_sc=0 }
  }
' "${CONFIG_PATH}")"

if [[ -z "${SCEN_PAIRS}" ]]; then
  echo "[ERR] Không tìm thấy experiments.scenarios trong ${CONFIG_PATH}." >&2
  exit 1
fi

# Đọc experiments.max_epochs và experiments.cap_per_class (mặc định).
CFG_MAX_EPOCHS="$(awk '/^  max_epochs:/ {print $2}' "${CONFIG_PATH}")"
CFG_CAP="$(awk '/^  cap_per_class:/ {print $2}' "${CONFIG_PATH}")"
CFG_PROTOCOLS="$(awk '/^  protocols:/ {
  $1=""; sub(/^[[:space:]]+/, "");
  gsub(/[\[\]]/, "");
  print
}' "${CONFIG_PATH}")"

# Apply override.
EPOCHS="${OVERRIDE_EPOCHS:-${CFG_MAX_EPOCHS}}"
CAP="${OVERRIDE_CAP:-${CFG_CAP}}"
PROTOCOLS="${CFG_PROTOCOLS:-per_scenario pooled loso}"

echo "  config             : ${CONFIG_PATH}"
echo "  scenarios          :"
echo "${SCEN_PAIRS}" | sed 's/^/    /'
echo "  protocols          : ${PROTOCOLS}"
echo "  max_epochs         : ${EPOCHS}"
echo "  cap_per_class      : ${CAP}"

# Sanity check: mọi scenario phải tồn tại trên đĩa.
echo
echo "  --- Kiểm tra file tồn tại ---"
MISSING=0
while IFS= read -r pair; do
  [[ -z "${pair}" ]] && continue
  path="${pair#*=}"
  if [[ ! -f "${path}" ]]; then
    echo "    [MISSING] ${path}"
    MISSING=$((MISSING + 1))
  else
    sz="$(du -h "${path}" | cut -f1)"
    echo "    [OK]      ${path}  (${sz})"
  fi
done <<< "${SCEN_PAIRS}"

if [[ "${MISSING}" -gt 0 ]]; then
  echo
  echo "[ERR] ${MISSING} file scenario bị THIẾU trên đĩa."
  echo "       Chạy: bash scripts/download_all.sh --apply"
  exit 3
fi

# Build tham số --scenarios cho python.
SCEN_ARGS=""
while IFS= read -r pair; do
  [[ -z "${pair}" ]] && continue
  if [[ -z "${SCEN_ARGS}" ]]; then
    SCEN_ARGS="${pair}"
  else
    SCEN_ARGS="${SCEN_ARGS} ${pair}"
  fi
done <<< "${SCEN_PAIRS}"

echo
echo "  → Sẽ chạy:"
echo "    python -m src.run_experiments \\"
echo "      --config ${CONFIG_PATH} \\"
echo "      --auto-resume \\"
echo "      --protocols ${PROTOCOLS} \\"
echo "      --scenarios ${SCEN_ARGS} \\"
echo "      --cap-per-class ${CAP} \\"
echo "      --epochs ${EPOCHS}"
echo

# -------- 3) Chạy orchestrator --------
echo "============================================================"
echo " STEP 3  ·  Run orchestrator (python -m src.run_experiments)"
echo "============================================================"
echo

set +e   # tạm tắt để bắt exit code; vẫn giữ set -u
python -m src.run_experiments \
  --config "${CONFIG_PATH}" \
  --auto-resume \
  --protocols ${PROTOCOLS} \
  --scenarios ${SCEN_ARGS} \
  --cap-per-class "${CAP}" \
  --epochs "${EPOCHS}"
PY_EXIT=$?
set -e

echo
echo "============================================================"
echo " STEP 4  ·  Orchestrator exit code = ${PY_EXIT}"
echo "============================================================"

if [[ "${PY_EXIT}" -ne 0 ]]; then
  echo "  [WARN] python thoát với exit code ${PY_EXIT}."
  echo "         Có thể instance bị OOM hoặc có lỗi trong training."
  echo "         results_summary.csv hiện có CHỨA kết quả đến trước khi lỗi"
  echo "         (đã save sau mỗi protocol) → có thể chạy lại --auto-resume."
fi

# -------- 4) Git backup (best-effort) --------
if [[ "${DO_GIT_BACKUP}" -eq 0 ]]; then
  echo "  --no-git → bỏ qua git backup."
  echo "============================================================"
  exit "${PY_EXIT}"
fi

echo
echo "============================================================"
echo " STEP 4  ·  Git backup artifacts/"
echo "============================================================"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "  [INFO] Không phải git repo (không có .git) → bỏ qua git backup."
  echo "         Để backup thủ công: rsync -av artifacts/ <remote>:<path>/"
  echo "============================================================"
  exit "${PY_EXIT}"
fi

echo "  [INFO] Repo git: $(git rev-parse --show-toplevel)"
git add artifacts/ 2>&1 | sed 's/^/    /' || true

if git diff --cached --quiet; then
  echo "  [INFO] Không có gì thay đổi trong artifacts/ — không commit."
else
  MSG="experiments: auto-backup $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if git commit -m "${MSG}" 2>&1 | sed 's/^/    /'; then
    echo "  [INFO] Đã commit."
  else
    echo "  [WARN] git commit thất bại — xem log trên."
    exit "${PY_EXIT}"
  fi

  echo "  --- git push ---"
  PUSH_LOG="$(mktemp)"
  if git push 2>&1 | tee "${PUSH_LOG}"; then
    echo "  [INFO] git push OK."
  else
    echo "  [WARN] git push FAILED (no credential? no remote?)."
    echo
    echo "  ┌────────────────────────────────────────────────────────────"
    echo "  │ FALLBACK: backup thủ công bằng rsync/scp:"
    echo "  │"
    echo "  │   rsync -avz artifacts/ user@backup.example.com:/path/to/backup/"
    echo "  │   # hoặc:"
    echo "  │   scp -r artifacts/ user@backup.example.com:/path/to/backup/"
    echo "  │"
    echo "  │ Hoặc nếu muốn thử lại git push sau khi setup SSH key:"
    echo "  │   git remote set-url origin git@github.com:USER/REPO.git"
    echo "  │   git push"
    echo "  └────────────────────────────────────────────────────────────"
  fi
  rm -f "${PUSH_LOG}"
fi

echo "============================================================"
exit "${PY_EXIT}"