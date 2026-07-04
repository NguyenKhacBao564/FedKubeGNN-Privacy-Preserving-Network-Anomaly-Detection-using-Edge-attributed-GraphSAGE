#!/usr/bin/env bash
# =============================================================
# scripts/download_all.sh
# Tải conn.log.labeled cho MỌI scenario khai báo trong config.yaml
# (block experiments.scenarios). Tự động chọn cách tải theo loại:
#
#   1. IoT-23 MALWARE capture (URL dạng
#      https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/...)
#      → ủy quyền cho scripts/download_data.sh (đã có logic tự tìm subdir
#        'bro/' / 'zeek/', tự ưu tiên file plain, tự giải nén).
#
#   2. IoT-23 BENIGN capture (URL dạng
#      https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/CTU-IoT-Benign-Capture/<Name>/...)
#      → hàm riêng ``_download_benign`` bên dưới (cấu trúc khác).
#
# Cả 2 đều IDEMPOTENT (bỏ qua file đã có), in dung lượng mỗi file, cảnh báo
# nếu tổng > 15 GB.
#
# Chế độ chạy:
#   bash scripts/download_all.sh                 # DRY-RUN (chỉ in kế hoạch)
#   bash scripts/download_all.sh --apply         # tải thật
#
# Lưu ý: KHÔNG tải file pcap (nặng GB), chỉ tải conn.log.labeled.
# =============================================================

set -euo pipefail

# -------- Args --------
DRY_RUN=1
DEST_DIR="data"
CONFIG_PATH="config.yaml"
WARN_GB=15          # cảnh báo nếu tổng > 15 GB

usage() {
  cat <<EOF
Cách dùng:
  bash scripts/download_all.sh                  # DRY-RUN (in kế hoạch)
  bash scripts/download_all.sh --apply          # tải thật
  bash scripts/download_all.sh --apply --config PATH
                                                 # đổi đường dẫn config (mặc định: config.yaml)
  bash scripts/download_all.sh --apply --dest DIR
                                                 # đổi thư mục đích (mặc định: ./data)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)        DRY_RUN=0 ;;
    --config)       CONFIG_PATH="$2"; shift ;;
    --dest)         DEST_DIR="$2"; shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "[ERR] Tham số lạ: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

# -------- Tool check --------
command -v wget >/dev/null 2>&1 || { echo "[ERR] Thiếu 'wget'. Cài: apt install wget (hoặc brew install wget)" >&2; exit 1; }
command -v grep >/dev/null 2>&1 || { echo "[ERR] Thiếu 'grep'." >&2; exit 1; }
command -v awk >/dev/null 2>&1  || { echo "[ERR] Thiếu 'awk'." >&2; exit 1; }
command -v du  >/dev/null 2>&1  || { echo "[ERR] Thiếu 'du'." >&2; exit 1; }

# -------- Helpers --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DATA_SH="${SCRIPT_DIR}/download_data.sh"
[[ -f "${DOWNLOAD_DATA_SH}" ]] || { echo "[ERR] Không tìm thấy ${DOWNLOAD_DATA_SH}" >&2; exit 1; }

MALWARE_BASE="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"
BENIGN_BASE="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/CTU-IoT-Benign-Capture"

# Phân loại scenario theo URL dựa vào ``name``:
#   - có chứa "Benign" hoặc nằm trong whitelist benign_names → dùng BENIGN_BASE
#   - ngược lại → dùng MALWARE_BASE
#
# Whitelist tên benign capture phổ biến trong IoT-23 (có thể mở rộng).
BENIGN_NAMES_REGEX='^(Somfy|Hue|Aria|Eero|Tuya|Wink|SmartThings|Philips_LED|Amazon_Echo|bose|Belkin|Netatmo|Sengled|Yi_Cam|iHome|Casino_Lights|Yale_Doorbell)'

_is_benign_name() {
  local name="$1"
  if [[ "${name}" == *Benign* || "${name}" == *benign* ]]; then
    return 0
  fi
  if [[ "${name}" =~ ${BENIGN_NAMES_REGEX} ]]; then
    return 0
  fi
  return 1
}

# ---- Parser moved inline (1-line gộp name+path) — see SCEN_LINES below. ----

# Tải 1 benign capture (cấu trúc khác với malware).
# Input: name, url path dạng "CTU-IoT-Benign-Capture/Somfy-..."
# Logic: thử index URL, tìm file chứa 'conn.log.labeled', tải về <DEST>/<name>/.
_download_benign() {
  local name="$1"
  local url_path="$2"   # ví dụ: "CTU-IoT-Benign-Capture/Somfy-1-1"
  local target_dir="${DEST_DIR}/${name}"
  local index_url="${BENIGN_BASE}/${url_path}/"
  local final_file="${target_dir}/conn.log.labeled"

  mkdir -p "${target_dir}"

  if [[ -f "${final_file}" ]] && [[ -s "${final_file}" ]]; then
    echo "    [SKIP] Đã tồn tại: ${final_file}"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "    [DRY-RUN] wget -q -O <index.html> \"${index_url}\""
    echo "    [DRY-RUN] grep -oE 'href=\"[^\"]*conn\\.log\\.labeled[^\"]*\"' <index.html>"
    echo "    [DRY-RUN] wget -q -O <out> \"<link>\" (ưu tiên file plain, .tar.gz, .gz)"
    return 0
  fi

  # ---- APPLY ----
  local tmp_idx
  tmp_idx="$(mktemp)"
  if ! wget -q -O "${tmp_idx}" "${index_url}"; then
    echo "    [WARN] Không tải được index của ${name}: ${index_url}" >&2
    rm -f "${tmp_idx}"
    return 1
  fi

  # Tìm link đầu tiên chứa 'conn.log.labeled'
  local found
  found="$(grep -oE 'href="[^"]*conn\.log\.labeled[^"]*"' "${tmp_idx}" \
          | sed -E 's/^href="([^"]+)"$/\1/' | sort -u || true)"
  rm -f "${tmp_idx}"

  if [[ -z "${found}" ]]; then
    echo "    [WARN] ${name}: không thấy link conn.log.labeled trong ${index_url}" >&2
    echo "    [HINT] Mở URL để xem cấu trúc thật, hoặc thêm tên vào blacklist." >&2
    return 1
  fi

  # Ưu tiên file plain (không .gz) > .tar.gz > .gz.
  # Regex PHẢI anchor `^` — nếu không thì `sort -u` Alphabet sẽ xếp
  # bản "<scenario>.conn.log.labeled" (có prefix) LÊN TRƯỚC `conn.log.labeled`
  # thuần → `head -n 1` chọn NHẦM file có prefix.
  local link=""
  link="$(echo "${found}" | grep -E '^/?conn\.log\.labeled(\.(tar\.)?gz)?$' | head -n 1 || true)"
  [[ -z "${link}" ]] && link="$(echo "${found}" | grep -E '\.tar\.gz$' | head -n 1 || true)"
  [[ -z "${link}" ]] && link="$(echo "${found}" | grep -E '\.gz$' | head -n 1 || true)"
  [[ -z "${link}" ]] && link="$(echo "${found}" | head -n 1)"

  # Ghép URL tuyệt đối nếu là tương đối
  if [[ "${link}" != http* ]]; then
    link="${index_url%/}/${link}"
  fi

  local out_file="${target_dir}/$(basename "${link}")"
  echo "    [GET ] ${link}"
  if ! wget -q --show-progress -O "${out_file}" "${link}"; then
    echo "    [WARN] Tải thất bại: ${link}" >&2
    rm -f "${out_file}"
    return 1
  fi

  # Giải nén nếu cần
  case "${out_file}" in
    *.tar.gz)
      echo "    [UNTAR] ${out_file}"
      tar -xzf "${out_file}" -C "${target_dir}"
      rm -f "${out_file}"
      ;;
    *.gz)
      echo "    [GZIP] ${out_file}"
      gunzip -f "${out_file}"
      ;;
  esac

  if [[ -f "${final_file}" ]]; then
    local sz
    sz="$(du -h "${final_file}" | cut -f1)"
    echo "    [SIZE] ${final_file}  (${sz})"
  else
    # In cả ``name`` + ``final_file`` + ``out_file`` (file đã tải về) để
    # tránh nhầm scenario khi log nhiều cái liên tiếp.
    echo "    [WARN] [${name}] Không thấy ${final_file} sau khi xử lý — kiểm tra thủ công." >&2
    echo "    [WARN] [${name}] Đã tải về: ${out_file:-<không có>}" >&2
    return 1
  fi
  return 0
}

# -------- Đọc config.yaml (rất simple parser) --------
[[ -f "${CONFIG_PATH}" ]] || { echo "[ERR] Không tìm thấy config: ${CONFIG_PATH}" >&2; exit 1; }

# Trích block experiments.scenarios: gộp mỗi scenario thành 1 dòng
# ``name=VALUE<TAB>path=VALUE`` để script bash đọc từng dòng dễ dàng
# (không cần xử lý nhiều dòng liên tiếp).
SCEN_LINES="$(awk '
  /^experiments:/ { in_exp=1; next }
  in_exp && /^  scenarios:/ { in_sc=1; next }
  in_sc {
    if (/^    - name:/) {
      # Lấy giá trị name (phần sau "name:" đến hết dòng, lstrip whitespace).
      val = $0; sub(/^[[:space:]]*-[[:space:]]*name:[[:space:]]*/, "", val);
      # Đọc dòng tiếp theo (path: ...).
      if ((getline p_line) > 0) {
        p_val = p_line; sub(/^[[:space:]]*path:[[:space:]]*/, "", p_val);
        print val "\t" p_val;
      }
      next;
    }
    if (/^  [a-z]/) { in_sc=0 }
  }
' "${CONFIG_PATH}")"

if [[ -z "${SCEN_LINES}" ]]; then
  echo "[ERR] Không tìm thấy experiments.scenarios trong ${CONFIG_PATH}." >&2
  exit 1
fi

# -------- Main --------
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "============================================================"
  echo " IoT-23 downloader — ALL scenarios (từ config.yaml)"
  echo " DEST_DIR = ${DEST_DIR}/<scenario>/"
  echo " MODE     = DRY-RUN (chỉ IN kế hoạch, KHÔNG tải thật)"
  echo "============================================================"
else
  echo "============================================================"
  echo " IoT-23 downloader — ALL scenarios (từ config.yaml)"
  echo " DEST_DIR = ${DEST_DIR}/<scenario>/"
  echo " MODE     = APPLY (đang tải thật)"
  echo "============================================================"
fi

TOTAL_BYTES=0
COUNT_OK=0
COUNT_FAIL=0
COUNT_SKIP=0
COUNT_BENIGN=0
COUNT_MALWARE=0

while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  # SCEN_LINES format: ``name\tpath`` (1 dòng / scenario).
  name="${line%%$'\t'*}"
  path="${line##*$'\t'}"
  [[ -z "${name}" || -z "${path}" ]] && continue

  echo
  echo "--- Scenario: ${name}"
  echo "    path in config: ${path}"

  # Xác định phân loại & URL dựa trên tên
  if _is_benign_name "${name}"; then
    COUNT_BENIGN=$((COUNT_BENIGN + 1))
    # path dạng "data/CTU-IoT-Benign-Capture/<Name>-..." → lấy phần tương đối
    rel="${path#data/}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "    [TYPE] BENIGN → base ${BENIGN_BASE}"
      echo "    [DRY-RUN] expected final path: ${path}"
    fi
    if _download_benign "${name}" "${rel}"; then
      COUNT_OK=$((COUNT_OK + 1))
      if [[ -f "${path}" ]]; then
        sz="$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}")"
        TOTAL_BYTES=$((TOTAL_BYTES + sz))
      fi
    else
      COUNT_FAIL=$((COUNT_FAIL + 1))
    fi
  else
    COUNT_MALWARE=$((COUNT_MALWARE + 1))
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "    [TYPE] MALWARE → sẽ ủy quyền scripts/download_data.sh --apply"
      echo "    [DRY-RUN] expected final path: ${path}"
    fi
    # Ủy quyền cho download_data.sh. Tạm thời file đã có sẽ được skip.
    if [[ "${DRY_RUN}" == "0" ]]; then
      bash "${DOWNLOAD_DATA_SH}" --apply --dest "${DEST_DIR}" >/dev/null
    fi
    if [[ -f "${path}" ]]; then
      sz="$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}")"
      TOTAL_BYTES=$((TOTAL_BYTES + sz))
      COUNT_OK=$((COUNT_OK + 1))
    else
      COUNT_FAIL=$((COUNT_FAIL + 1))
    fi
  fi
done <<< "${SCEN_LINES}"

# -------- Tổng kết --------
echo
echo "============================================================"
echo " Total bytes   : ${TOTAL_BYTES}  ($(echo "scale=2; ${TOTAL_BYTES}/1024/1024/1024" | bc 2>/dev/null || echo "${TOTAL_BYTES}") GB)"
echo " Scenarios OK   : ${COUNT_OK}"
echo " Scenarios FAIL : ${COUNT_FAIL}"
echo "   - malware   : ${COUNT_MALWARE}"
echo "   - benign    : ${COUNT_BENIGN}"

GB_LIMIT=$((WARN_GB * 1024 * 1024 * 1024))
if [[ "${TOTAL_BYTES}" -gt "${GB_LIMIT}" ]]; then
  echo
  echo "  ⚠ CẢNH BÁO: Tổng dung lượng ${TOTAL_BYTES} bytes > ${WARN_GB} GB."
  echo "    Nếu disk GPU vast.ai nhỏ, cân nhắc cap_per_class nhỏ hơn."
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo " DRY-RUN xong. Không có file nào được tải."
  echo " Để chạy thật: bash scripts/download_all.sh --apply"
else
  if [[ "${COUNT_FAIL}" -gt 0 ]]; then
    echo
    echo " ⚠ ${COUNT_FAIL} scenario KHÔNG tải được — xem [WARN] phía trên."
    echo "   Có thể cần tải thủ công hoặc đổi URL."
    exit 2
  fi
  echo
  echo " Hoàn tất. File nằm trong ${DEST_DIR}/<scenario>/conn.log.labeled"
fi
echo "============================================================"