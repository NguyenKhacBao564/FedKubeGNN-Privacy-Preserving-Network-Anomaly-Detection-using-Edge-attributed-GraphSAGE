#!/usr/bin/env bash
# =============================================================
# scripts/download_data.sh
# Tải file conn.log.labeled của 6 scenario IoT-23 đã chọn cho GĐ1.
#
# Vì đường dẫn con trong từng scenario KHÔNG hoàn toàn đồng nhất
# (có thể nằm trong 'bro/', 'zeek/', hoặc thư mục khác), script sẽ:
#   1. Tải trang index của scenario (HTML directory listing).
#   2. Tìm link tới file có tên chứa 'conn.log.labeled'
#      (ưu tiên .gz / .tar.gz nếu có, vì file đã nén).
#   3. Tải file đó về. Nếu nén thì giải nén.
#
# CHẾ ĐỘ CHẠY (mặc định an toàn):
#   bash scripts/download_data.sh           # DRY-RUN: chỉ IN lệnh wget dự định
#   bash scripts/download_data.sh --apply   # APPLY: thực sự tải xuống
#
# LƯU Ý:
#   • Chỉ tải conn.log.labeled, KHÔNG tải pcap (pcap nặng GB, không dùng).
#   • Bản v1 (khớp lý thuyết đã chuẩn bị), không dùng v2.
#   • Nguồn: mcfp.felk.cvut.cz — kho public chính thức của IoT-23.
#   • Idempotent: file đã tồn tại thì bỏ qua, không tải lại.
# =============================================================

set -euo pipefail

# -------- Args --------
DRY_RUN=1            # 1 = chỉ in lệnh, 0 = tải thật
DEST_DIR="data"
INDEX_DIR=""          # thư mục tạm để cache index HTML (APPLY mới cần)

usage() {
  cat <<EOF
Cách dùng:
  bash scripts/download_data.sh                # DRY-RUN (mặc định)
  bash scripts/download_data.sh --apply        # tải thật
  bash scripts/download_data.sh --apply --dest DIR
                                                # đổi thư mục đích (mặc định: ./data)
  bash scripts/download_data.sh -h | --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)        DRY_RUN=0 ;;
    --dest)         DEST_DIR="$2"; shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "[ERR] Tham số lạ: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

# -------- Tool check --------
command -v wget >/dev/null 2>&1 || { echo "[ERR] Thiếu 'wget'. Cài: brew install wget" >&2; exit 1; }
command -v grep >/dev/null 2>&1 || { echo "[ERR] Thiếu 'grep'." >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo "[ERR] Thiếu 'tar'." >&2; exit 1; }

# -------- Config --------
SCENARIOS=(
  "CTU-IoT-Malware-Capture-34-1"
  "CTU-IoT-Malware-Capture-1-1"
  "CTU-IoT-Malware-Capture-3-1"
  "CTU-IoT-Malware-Capture-9-1"
  "CTU-IoT-Malware-Capture-36-1"
  "CTU-IoT-Malware-Capture-39-1"
)

BASE_URL="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "============================================================"
  echo " IoT-23 downloader — GĐ1 (6 scenario)"
  echo " DEST_DIR = ${DEST_DIR}/<scenario>/"
  echo " MODE     = DRY-RUN (chỉ IN lệnh wget, KHÔNG tải thật)"
  echo "============================================================"
else
  echo "============================================================"
  echo " IoT-23 downloader — GĐ1 (6 scenario)"
  echo " DEST_DIR = ${DEST_DIR}/<scenario>/"
  echo " MODE     = APPLY (đang tải thật)"
  echo "============================================================"
  INDEX_DIR="$(mktemp -d -t iot23_index_XXXXXX)"
  trap 'rm -rf "${INDEX_DIR}"' EXIT
fi

# -------- Helper: phát hiện link tới conn.log.labeled trong HTML index --------
# Trả về link đầu tiên khớp, ưu tiên .gz > .tar.gz > raw. Nếu không thấy → rỗng.
find_conn_log_link() {
  local html_file="$1"
  local base="$2"   # URL gốc của index (không có trailing slash cũng được)
  # Tìm mọi href chứa 'conn.log.labeled'
  local found
  found="$(grep -oE 'href="[^"]*conn\.log\.labeled[^"]*"' "${html_file}" \
          | sed -E 's/^href="([^"]+)"$/\1/' \
          | sort -u)"

  if [[ -z "${found}" ]]; then
    return 1
  fi

  # Ưu tiên file "plain" (không có prefix tên scenario phía trước) — đây
  # là file đúng định dạng Zeek cần cho parser. Một số scenario (vd 1-1)
  # có thêm bản "<tên_scenario>.conn.log.labeled" mà ta KHÔNG dùng.
  #
  # Regex PHẢI anchor `^` để chỉ match basename bắt đầu bằng `conn.log.labeled`
  # (cho phép thêm 1 dấu `/` ở đầu). Không anchor thì `sort -u` xếp Alphabet
  # sẽ đưa bản có prefix ("CTU-IoT-Malware-1-1.conn.log.labeled") lên trước
  # `conn.log.labeled` thuần → `head -n 1` chọn NHẦM file có prefix.
  local link=""
  link="$(echo "${found}" | grep -E '^/?conn\.log\.labeled(\.(tar\.)?gz)?$' | head -n 1 || true)"
  if [[ -z "${link}" ]]; then
    # Fallback: vẫn ưu tiên .tar.gz > .gz > không nén cho các file có prefix
    link="$(echo "${found}" | grep -E '\.tar\.gz$' | head -n 1 || true)"
  fi
  if [[ -z "${link}" ]]; then
    link="$(echo "${found}" | grep -E '\.gz$' | head -n 1 || true)"
  fi
  if [[ -z "${link}" ]]; then
    link="$(echo "${found}" | grep -vE '\.(tar\.)?gz$' | head -n 1 || true)"
  fi

  if [[ -z "${link}" ]]; then
    return 1
  fi

  # Nếu link là tương đối, ghép với base
  if [[ "${link}" != http* ]]; then
    link="${base%/}/${link}"
  fi
  echo "${link}"
}


# -------- Helper: tìm subdir chứa conn.log.labeled --------
# Vì hầu hết scenario IoT-23 đặt file trong 'bro/' (Apache directory listing
# cho phép duyệt từng thư mục), ta thử fetch index của một vài subdir phổ biến.
# Nếu subdir đó có chứa 'conn.log.labeled' → trả về URL subdir đó.
find_subdir_with_conn_log() {
  local base="$1"   # URL index của scenario (không trailing slash)
  for sub in bro zeek; do
    local sub_url="${base}/${sub}/"
    local tmp_idx
    tmp_idx="$(mktemp)"
    if wget -q -O "${tmp_idx}" "${sub_url}" 2>/dev/null \
       && grep -q 'conn\.log\.labeled' "${tmp_idx}" 2>/dev/null; then
      rm -f "${tmp_idx}"
      echo "${sub_url}"
      return 0
    fi
    rm -f "${tmp_idx}"
  done
  return 1
}

# -------- Main loop --------
for SCEN in "${SCENARIOS[@]}"; do
  TARGET_DIR="${DEST_DIR}/${SCEN}"
  mkdir -p "${TARGET_DIR}"

  INDEX_URL="${BASE_URL}/${SCEN}/"
  echo
  echo "--- Scenario: ${SCEN}"
  echo "    Index URL : ${INDEX_URL}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "    [DRY-RUN] wget -q -O <index.html> \"${INDEX_URL}\""
    echo "    [DRY-RUN] grep -oE 'href=\"[^\"]*conn\\.log\\.labeled[^\"]*\"' <index.html> | sort -u"
    echo "    [DRY-RUN] (ưu tiên file 'conn.log.labeled' thuần, không prefix)"
    echo "    [DRY-RUN] Nếu KHÔNG thấy ở root → thử subdir 'bro/' rồi 'zeek/'."
    echo "    [DRY-RUN] wget -q -O <index.html> \"${INDEX_URL}bro/\""
    echo "    [DRY-RUN] wget -q -O <out_file> \"<link tìm được>\""
    echo "    [DRY-RUN] (nếu nén: tar -xzf ... hoặc gunzip -f ...)"
    continue
  fi

  # ---- APPLY ----
  INDEX_FILE="${INDEX_DIR}/${SCEN}.html"
  if ! wget -q -O "${INDEX_FILE}" "${INDEX_URL}"; then
    echo "    [WARN] Không tải được index của ${SCEN}. Bỏ qua." >&2
    continue
  fi

  DL_LINK="$(find_conn_log_link "${INDEX_FILE}" "${INDEX_URL}" || true)"

  # Nếu không thấy ở root, thử tìm trong subdir phổ biến (bro/, zeek/).
  if [[ -z "${DL_LINK}" ]]; then
    SUB_URL="$(find_subdir_with_conn_log "${INDEX_URL%/}" || true)"
    if [[ -n "${SUB_URL}" ]]; then
      echo "    [HINT] Không thấy ở root — tìm thấy trong subdir: ${SUB_URL}"
      SUB_INDEX="${INDEX_DIR}/${SCEN}_sub.html"
      wget -q -O "${SUB_INDEX}" "${SUB_URL}"
      DL_LINK="$(find_conn_log_link "${SUB_INDEX}" "${SUB_URL}" || true)"
    fi
  fi

  if [[ -z "${DL_LINK}" ]]; then
    echo "    [WARN] Không tìm thấy link conn.log.labeled trong index của ${SCEN}." >&2
    echo "    [HINT] Mở URL sau để xem cấu trúc thật: ${INDEX_URL}" >&2
    continue
  fi

  OUT_FILE="${TARGET_DIR}/$(basename "${DL_LINK}")"
  if [[ -f "${OUT_FILE}" ]]; then
    echo "    [SKIP] Đã tồn tại: ${OUT_FILE}"
  else
    echo "    [GET ] ${DL_LINK}"
    if ! wget -q --show-progress -O "${OUT_FILE}" "${DL_LINK}"; then
      echo "    [WARN] Tải thất bại: ${DL_LINK}" >&2
      continue
    fi
  fi

  # ---- Giải nén nếu cần ----
  case "${OUT_FILE}" in
    *.tar.gz)
      echo "    [UNTAR] ${OUT_FILE}"
      tar -xzf "${OUT_FILE}" -C "${TARGET_DIR}"
      rm -f "${OUT_FILE}"
      ;;
    *.gz)
      echo "    [GZIP] ${OUT_FILE}"
      gunzip -f "${OUT_FILE}"
      ;;
  esac

  FINAL_FILE="${TARGET_DIR}/conn.log.labeled"
  if [[ -f "${FINAL_FILE}" ]]; then
    SIZE="$(du -h "${FINAL_FILE}" | cut -f1)"
    echo "    [SIZE] ${FINAL_FILE}  (${SIZE})"
  else
    # In cả SCEN + FINAL_FILE để tránh nhầm khi log nhiều scenario.
    echo "    [WARN] [${SCEN}] Không thấy ${FINAL_FILE} sau khi xử lý — kiểm tra thủ công." >&2
    echo "    [WARN] [${SCEN}] Đã tải về: ${OUT_FILE:-<không có>}" >&2
  fi
done

echo
echo "============================================================"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo " DRY-RUN xong. Không có file nào được tải."
  echo " Để chạy thật: bash scripts/download_data.sh --apply"
else
  echo " Hoàn tất. File nằm trong ${DEST_DIR}/<scenario>/conn.log.labeled"
fi
echo "============================================================"