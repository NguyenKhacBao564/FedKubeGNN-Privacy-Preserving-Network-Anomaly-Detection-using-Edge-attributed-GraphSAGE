"""
data_io.py — Đọc file conn.log.labeled (định dạng Zeek TSV) → pandas.DataFrame.

Triển khai cho Giai đoạn 1 (Task 1.2-1.4):
    1. Tự parse dòng '#fields' để lấy tên cột (KHÔNG hardcode).
    2. Đọc dữ liệu với sep='\\t', comment='#' để bỏ mọi dòng metadata.
    3. Tách cột cuối (đang gộp 3 giá trị cách nhau bằng SPACE —
       'tunnel_parents label detailed-label') thành 3 cột riêng.
    4. KHÔNG tự chuyển '-' thành NaN ở bước này — để bước preprocess xử lý.
    5. Module này KHÔNG xử lý missing / encoding / scale (đó là preprocess.py)
       và KHÔNG dựng đồ thị (đó là graph_build.py).

Reproducibility: KHÔNG shuffle ở đây. Shuffle chỉ làm ở tầng train/test split.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import List, Optional, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------

def read_conn_log(path: str) -> pd.DataFrame:
    """
    Đọc 1 file conn.log.labeled (định dạng Zeek TSV) → DataFrame.

    Bước xử lý:
        1. Mở file, tìm dòng bắt đầu bằng '#fields', tách theo TAB để lấy
           tên các cột. KHÔNG hardcode tên cột.
        2. Đọc phần data với sep='\\t', comment='#' để bỏ mọi dòng metadata
           ('#separator', '#types', '#close', ...).
        3. KHÔNG tự chuyển '-' thành NaN — đọc tất cả thành chuỗi để giữ
           nguyên giá trị thô (bước preprocess sẽ lo).

    Lưu ý về "lỗi định dạng" của IoT-23:
        • File conn.log.labeled có dòng '#fields' chứa 21 tên cột, nhưng tên
          CUỐI CÙNG là một chuỗi có 3 tên con cách nhau bằng khoảng trắng:
          'tunnel_parents   label   detailed-label'. Đây là do Zeek gốc chỉ
          có cột `tunnel_parents`, IoT-23 append thêm 2 cột nhãn nhưng ghi
          chung vào một ô header.
        • Trong DỮ LIỆU, 3 giá trị này là 3 TRƯỜNG TAB-SEPARATED riêng biệt
          (không phải gộp trong 1 ô như tên cột). Tức là mỗi dòng data có
          21 + 2 = 23 trường khi đọc bằng sep='\\t'.
        • Hàm này tự động phát hiện và tách tên cột cuối để khớp với data.
          Nếu dữ liệu thật lại dùng định dạng "gộp trong 1 ô" (3 giá trị
          space-separated), split_label_column() sẽ xử lý tiếp.

    Parameters
    ----------
    path : str
        Đường dẫn tới file conn.log.labeled.

    Returns
    -------
    pd.DataFrame
        DataFrame với các cột theo thứ tự trong '#fields' (đã tách tên cột
        cuối nếu bị gộp). Tất cả cột kiểu str, KHÔNG tự chuyển '-' → NaN.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    # 1. Tìm dòng '#fields' để lấy tên cột.
    field_names: Optional[List[str]] = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#fields"):
                # Tách theo TAB: phần tử đầu là chuỗi '#fields', phần còn lại là tên cột.
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    raise ValueError(
                        f"Dòng #fields trong {path} không hợp lệ: {line!r}"
                    )
                field_names = parts[1:]
                break

    if field_names is None:
        raise ValueError(
            f"Không tìm thấy dòng '#fields' trong file {path}. "
            "File có đúng định dạng Zeek không?"
        )

    logger.info("read_conn_log: %s — tìm thấy %d tên cột trong '#fields'.",
                path, len(field_names))

    # 2. Chuẩn hóa tên cột: IoT-23 không thống nhất giữa các scenario —
    # một số file dùng 'detailed-label', số khác dùng 'det_label' hoặc
    # 'detailed_label'. Đưa về dạng chuẩn để downstream code chỉ tham
    # chiếu 1 tên duy nhất.
    canonical_map = {
        "det_label": "detailed-label",
        "detailed_label": "detailed-label",
        "label_val": "label",
    }
    normalized = []
    renamed_any = False
    for name in field_names:
        canon = canonical_map.get(name, name)
        if canon != name:
            renamed_any = True
        normalized.append(canon)
    field_names = normalized
    if renamed_any:
        logger.info("read_conn_log: đã chuẩn hóa tên cột về dạng canonical.")

    # 3. Đọc data với sep=TAB, comment='#', KHÔNG tự chuyển '-' thành NaN.
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=field_names,
        na_values=[],            # tắt chuỗi nào cũng không auto-NaN
        keep_default_na=False,   # KHÔNG chuyển '-' / NaN literal thành NaN
        skip_blank_lines=True,
        dtype=str,               # giữ tất cả giá trị thô dạng chuỗi
        engine="python",         # cần thiết cho sep + comment kết hợp
        on_bad_lines="skip",     # bỏ dòng lỗi (nếu có)
    )

    logger.info("read_conn_log: đọc được %d dòng × %d cột.",
                df.shape[0], df.shape[1])
    return df


def split_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đảm bảo 3 cột cuối là 'tunnel_parents', 'label', 'detailed-label'
    với đúng giá trị. Hàm chịu 2 định dạng khác nhau của IoT-23:

    (a) Format A (5/6 scenario: 1-1, 3-1, 9-1, 36-1, 39-1):
        read_conn_log trả về 21 cột (vì #fields gộp 3 tên cuối). Cột cuối
        chứa 3 giá trị SPACE-separated kiểu "(empty)   Malicious
        PartOfAHorizontalPortScan". Hàm sẽ tách thành 3 cột riêng.

    (b) Format B (scenario 34-1):
        read_conn_log trả về 23 cột (đã tách sẵn). Hàm chỉ cần đảm bảo
        3 cột cuối đặt tên đúng 'tunnel_parents', 'label', 'detailed-label'.

    Sau khi xử lý, kiểm tra cột 'label': chỉ nên chứa {Benign, Malicious}
    (so sánh không phân biệt hoa/thường). Nếu có giá trị lạ → log warning
    (đây là dấu hiệu tách cột sai).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầu vào từ read_conn_log.

    Returns
    -------
    pd.DataFrame
        DataFrame với 3 cột cuối là 'tunnel_parents', 'label',
        'detailed-label'. Nếu đã đúng từ đầu, trả về bản sao. Nếu bị
        gộp, tách rồi trả về.
    """
    if df.shape[1] == 0:
        raise ValueError("split_label_column: DataFrame rỗng (0 cột).")

    df = df.copy()
    expected_last_3 = ["tunnel_parents", "label", "detailed-label"]
    last_3 = list(df.columns[-3:])
    last_col = df.columns[-1]

    # Trường hợp (b): đã có 3 cột cuối tên đúng → chỉ cần đảm bảo tên
    # đúng canonical và return.
    if last_3 == expected_last_3:
        logger.info("split_label_column: 3 cột cuối đã đặt tên đúng — no-op.")
        return df

    # Trường hợp (a): cột cuối chứa 3 giá trị space-separated → tách.
    def _split_one(val: object) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not isinstance(val, str):
            return (None, None, None)
        parts = re.split(r"\s+", val.strip(), maxsplit=2)
        while len(parts) < 3:
            parts.append(None)
        return (parts[0], parts[1], parts[2])

    split_tuples = df[last_col].apply(_split_one)
    split_df = pd.DataFrame(
        split_tuples.tolist(),
        columns=expected_last_3,
        index=df.index,
    )

    df = df.drop(columns=[last_col])
    df = pd.concat([df, split_df], axis=1)
    logger.info("split_label_column: đã tách cột gộp '%s' thành 3 cột.", last_col)

    # Kiểm tra cột 'label' hợp lệ (không phân biệt hoa/thường).
    label_values = df["label"].dropna().astype(str).str.strip()
    unique_labels = set(label_values.unique())
    expected = {"benign", "malicious"}
    actual_lower = {v.lower() for v in unique_labels}
    unexpected = actual_lower - expected
    if unexpected:
        logger.warning(
            "split_label_column: phát hiện giá trị 'label' ngoài "
            "{Benign, Malicious}: %s. Có thể là do tách cột sai hoặc "
            "định dạng dữ liệu khác.",
            sorted(unexpected),
        )

    return df


def load_scenario(path: str) -> pd.DataFrame:
    """
    Gộp read_conn_log + split_label_column cho 1 file conn.log.labeled.

    Returns
    -------
    pd.DataFrame
        DataFrame đã tách cột nhãn, sẵn sàng cho bước preprocess.
    """
    df = read_conn_log(path)
    df = split_label_column(df)
    return df


def quick_eda(df: pd.DataFrame) -> None:
    """
    In nhanh thông tin khám phá dữ liệu (Task 1.3) ra stdout:
        • Shape, danh sách cột.
        • Dtypes từng cột.
        • Số lượng '-', '(empty)', NaN theo từng cột.
        • Phân bố value_counts của 'label' và 'detailed-label'.

    Chú ý: vì read_conn_log KHÔNG chuyển '-' → NaN nên cột kiểu object sẽ
    có '-' là chuỗi bình thường; cần đếm riêng.
    """
    print("=" * 70)
    print(f"Shape: {df.shape[0]:,} dòng × {df.shape[1]} cột")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("=" * 70)

    print("\n[1] Dtypes:")
    print(df.dtypes.to_string())

    print("\n[2] Số lượng giá trị thiếu / rỗng theo cột:")
    counts = {}
    for col in df.columns:
        s = df[col]
        n_nan = int(s.isna().sum())
        if s.dtype == object:
            n_dash = int((s == "-").sum())
            n_empty = int((s == "(empty)").sum())
        else:
            n_dash = 0
            n_empty = 0
        counts[col] = {"NaN": n_nan, "'-'": n_dash, "'(empty)'": n_empty}
    counts_df = pd.DataFrame(counts).T
    print(counts_df.to_string())

    if "label" in df.columns:
        print("\n[3] Phân bố 'label':")
        print(df["label"].value_counts(dropna=False).to_string())

    if "detailed-label" in df.columns:
        print("\n[4] Phân bố 'detailed-label':")
        print(df["detailed-label"].value_counts(dropna=False).to_string())

    print("=" * 70)


# ---------------------------------------------------------------------------
# Mock test (chạy được trên máy Mac không có dataset thật — Task 1.2-1.4)
# ---------------------------------------------------------------------------

_MOCK_CONN_LOG = """\
#separator \\x09
#set_separator	,
#empty_field	(empty)
#unset_field	-
#path	conn.log
#open	2024-01-01-00-00-00
#fields	ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	duration	orig_bytes	resp_bytes	conn_state	local_orig	local_resp	missed_bytes	history	orig_pkts	orig_ip_bytes	resp_pkts	resp_ip_bytes	tunnel_parents   label   detailed-label
#types	time	string	addr	port	addr	port	enum	string	interval	count	count	string	bool	bool	count	string	count	count	count	count	set[string]   string   string
1704067200.123456	C1	192.168.1.10	54321	8.8.8.8	53	udp	dns	0.001	50	80	SF	-	-	0	Dd	1	78	1	78	- Benign Benign
1704067201.234567	C2	192.168.1.10	54322	8.8.4.4	53	udp	dns	0.002	40	120	SF	-	-	0	Dd	1	68	1	68	- Benign Benign
1704067202.345678	C3	192.168.1.10	4444	45.83.66.1	6667	tcp	-	10.5	200	300	S1	-	-	0	ShADadtaF	5	400	4	500	- Malicious C&C-Mirai
1704067203.456789	C4	192.168.1.10	5555	45.83.66.2	23	tcp	telnet	2.3	100	150	S1	-	-	0	ShADadtaF	3	200	2	250	- Malicious C&C-FileDownload
1704067204.567890	C5	192.168.1.10	33333	198.51.100.5	80	tcp	http	0.5	500	1500	SF	-	-	0	ShADadtaF	10	700	8	900	- Benign Benign
1704067205.678901	C6	192.168.1.10	33334	198.51.100.6	80	tcp	-	0.01	60	40	S0	-	-	0	Sh	1	88	1	88	- Malicious PartOfAHorizontalPortScan
1704067206.789012	C7	192.168.1.10	33335	198.51.100.7	22	tcp	ssh	1.5	300	800	SF	-	-	0	ShADadtaF	4	500	3	600	- Benign Benign
#close 2024-01-01-01-00-00
"""


def _create_mock_file(directory: str) -> str:
    """Ghi nội dung mock ra file tạm, trả về đường dẫn."""
    os.makedirs(directory, exist_ok=True)
    fp = os.path.join(directory, "mock_conn.log.labeled")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(_MOCK_CONN_LOG)
    return fp


def _run_mock_test() -> None:
    """Chạy read_conn_log + split_label_column + quick_eda trên file mock."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    tmp_dir = tempfile.mkdtemp(prefix="iot23_mock_")
    mock_path = _create_mock_file(tmp_dir)
    print(f"\n>>> Mock file: {mock_path}\n")

    print(">>> Bước 1: read_conn_log")
    df_raw = read_conn_log(mock_path)
    print(f"    Shape: {df_raw.shape}")
    print(f"    Columns: {list(df_raw.columns)}")
    print(f"    Cột cuối (còn gộp) — 3 dòng đầu:")
    for v in df_raw.iloc[:3, -1]:
        print(f"      {v!r}")
    print()

    print(">>> Bước 2: split_label_column")
    df = split_label_column(df_raw)
    print(f"    Shape sau tách: {df.shape}")
    print(f"    3 cột cuối (tunnel_parents / label / detailed-label):")
    print(df[["tunnel_parents", "label", "detailed-label"]].to_string(index=False))
    print()

    print(">>> Bước 3: load_scenario (gộp read + split)")
    df_full = load_scenario(mock_path)
    assert df_full.shape == df.shape, "load_scenario không khớp với read+split!"
    print(f"    OK — shape: {df_full.shape}\n")

    print(">>> Bước 4: quick_eda")
    quick_eda(df_full)

    # Sanity assertions.
    assert df_full.shape[0] == 7, f"Expected 7 rows, got {df_full.shape[0]}"
    assert "tunnel_parents" in df_full.columns
    assert "label" in df_full.columns
    assert "detailed-label" in df_full.columns
    assert df_full["label"].iloc[0] == "Benign"
    assert df_full["label"].iloc[2] == "Malicious"
    assert df_full["detailed-label"].iloc[2] == "C&C-Mirai"
    assert df_full["detailed-label"].iloc[3] == "C&C-FileDownload"
    # Đảm bảo '-' không bị chuyển thành NaN ở bước đọc.
    assert (df_full["local_orig"] == "-").sum() >= 1, \
        "Giá trị '-' phải được giữ nguyên sau read_conn_log."
    assert (df_full["service"] == "-").sum() >= 1, \
        "Giá trị '-' phải được giữ nguyên ở cột service."
    # tunnel_parents ở mock đều là '-'.
    assert (df_full["tunnel_parents"] == "-").all(), \
        "tunnel_parents ở mock đều phải là '-'."

    print("\n[MOCK TEST] Tất cả assertions đều PASS.")


if __name__ == "__main__":
    _run_mock_test()