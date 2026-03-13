"""
系統図PDF → 系統モデル（ノード・エッジ）自動生成パイプライン multi
対象: 東北電力ネットワーク 全7県系統図

pipeline_v2.py の汎用化版。以下を追加:
  1. 全7県PDFを一括処理 (REGIONS辞書で管理)
  2. rotation=90ページ (青森・福島) のテキスト座標変換
  3. 都道府県境界端点検出 (cross_prefecture フラグ)
  4. インセット領域ブリッジング (INSET_BRIDGE_R=150pt)

工程:
  A. 図形抽出 (PDF → 線分集合 + 設備記号)
  B. 送電線候補抽出 (66kV / 33kV 別フィルタ)
  C. 接続点同一点化 (DBSCAN スナップ)
  D. ネットワーク化
  E. 路線(line_id)割当 + CSV名称付与
  F. ノード確定 + 変電所種別分類
  G. 線路長算出
  H. ギャップブリッジング (bridge / provisional / inset_provisional)
  I. 都道府県境界端点検出
  J. 可視化
"""

import fitz           # PyMuPDF
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定 (Windowsの場合)
for _fname in ['MS Gothic', 'Yu Gothic', 'Meiryo', 'IPAGothic']:
    try:
        fm.findfont(_fname, fallback_to_default=False)
        matplotlib.rcParams['font.family'] = _fname
        break
    except Exception:
        pass

import matplotlib.patches as mpatches
from shapely.geometry import LineString, Point  # noqa: F401 (将来利用)
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import json
import math
import os
import re
import sys
from collections import Counter

# ─────────────────────────────────────────────
# 地域設定
# ─────────────────────────────────────────────
# 地域別線幅パラメータ: 各PDFで送電線の描画線幅が異なるため地域ごとに設定
# (w66_min, w66_max): 66kV線幅範囲 [pt]
# (w33_min, w33_max): 33kV線幅範囲 [pt]
# 測定値: 01=1.50/0.50, 02=1.41/0.705, 03=1.50/0.75, 04=1.74/0.87,
#         05=1.65/0.55, 06=1.56/0.78, 07=1.275/0.637
REGIONS = {
    '01': {'name': '青森県', 'prefix': 'local01', 'w66': (1.35, 1.65), 'w33': (0.40, 0.60)},
    '02': {'name': '岩手県', 'prefix': 'local02', 'w66': (1.30, 1.55), 'w33': (0.60, 0.80)},
    '03': {'name': '秋田県', 'prefix': 'local03', 'w66': (1.35, 1.65), 'w33': (0.60, 0.85)},
    '04': {'name': '宮城県', 'prefix': 'local04', 'w66': (1.55, 1.95), 'w33': (0.75, 1.00)},
    '05': {'name': '山形県', 'prefix': 'local05', 'w66': (1.40, 1.90), 'w33': (0.45, 0.65)},
    '06': {'name': '福島県', 'prefix': 'local06', 'w66': (1.40, 1.75), 'w33': (0.65, 0.90)},
    '07': {'name': '新潟県', 'prefix': 'local07', 'w66': (1.10, 1.45), 'w33': (0.55, 0.75)},
}

BASE_DIR     = r"C:\Users\taku.kashiwazaki\Documents\PowerSystem"
INPUT_DIR    = os.path.join(BASE_DIR, "Input", "tohoku_local")
OUTPUT_BASE  = os.path.join(BASE_DIR, "output_multi")

# ─────────────────────────────────────────────
# 定数パラメータ (全地域共通)
# ─────────────────────────────────────────────
WIDTH_66KV_MIN    = 1.40   # pt
WIDTH_66KV_MAX    = 1.90   # pt
WIDTH_33KV_MIN    = 0.45   # pt
WIDTH_33KV_MAX    = 0.65   # pt
COLOR_DARK_THRESH = 0.15
SEG_LEN_MIN       = 3.0    # pt
EPS_SNAP          = 10.0   # DBSCAN eps (pt)
R_LABEL           = 50.0   # pt
R_ATTACH          = 12.0   # pt
GEN_R_ATTACH      = 30.0   # pt
GEN_FALLBACK_R    = 50.0   # pt
GEN_COLOR         = (0.4, 0.84, 1.0)
GEN_COLOR_TOL     = 0.10
BRIDGE_THRESH_REAL = 22.0  # pt → connection_type='bridge'
BRIDGE_THRESH_PROV = 60.0  # pt → connection_type='provisional'
INSET_BRIDGE_R    = 150.0  # pt → connection_type='inset_provisional'
BORDER_MARGIN     = 25.0   # pt from page edge
BEZIER_DIV        = 20
PAGE_NUM          = 0      # 0-indexed
SNAP_DIST_LINE    = 25.0   # pt: シンボルが送電線に吸着 (セグメント分割)
SNAP_FALLBACK     = 70.0   # pt: それ以上は仮想接続 (孤立ノード化)
BORDER_RECT_FRAC  = 0.55   # ページ面積この割合以上の矩形パスは外枠と判断
BORDER_LONG_FRAC  = 0.40   # ページ辺のこの割合以上の直線を端近傍でフィルタ
BORDER_EDGE_MARGIN= 50.0   # pt: ページ端からこの距離以内の長い直線を外枠と判断
N_PTS_SEG_MAX     = 60     # 折れ線点数がこれを超えると曲線(県境ライン)と判断


# ─────────────────────────────────────────────
# ユーティリティ関数
# ─────────────────────────────────────────────

def bezier_to_points(p0, p1, p2, p3, n=BEZIER_DIV):
    """3次ベジェ曲線 → n+1 個の点列"""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def seg_length(x0, y0, x1, y1):
    return math.sqrt((x1-x0)**2 + (y1-y0)**2)


def polyline_length(pts):
    total = 0.0
    for i in range(len(pts)-1):
        total += seg_length(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
    return total


def is_dark(color):
    if color is None:
        return False
    r, g, b = color[0], color[1], color[2]
    return r <= COLOR_DARK_THRESH and g <= COLOR_DARK_THRESH and b <= COLOR_DARK_THRESH


def is_dashed(dashes_attr):
    """
    PyMuPDFはdashes属性を文字列で返す。
    "[] 0"  → 実線 (dashなし)
    "[ x y ] 0" → 破線
    """
    if dashes_attr is None:
        return False
    s = str(dashes_attr).strip()
    if s in ('[] 0', '[]', '', 'None'):
        return False
    if s.startswith('['):
        inner = s.split(']')[0].replace('[', '').strip()
        if inner == '':
            return False
        return bool(re.search(r'\d', inner))
    return False


def is_gen_color(color):
    if color is None:
        return False
    return all(abs(color[i] - GEN_COLOR[i]) <= GEN_COLOR_TOL for i in range(3))


def path_to_segments(path):
    """
    1つのPDFパスオブジェクトを線分点列リストに変換する。
    戻り値: list of list of (x, y)
    """
    items = path.get('items', [])
    segments = []
    current = []

    for item in items:
        itype = item[0]
        if itype == 'l':
            p0 = (item[1].x, item[1].y)
            p1 = (item[2].x, item[2].y)
            if not current:
                current.append(p0)
            current.append(p1)
        elif itype == 'c':
            bp0 = (item[1].x, item[1].y)
            bp1 = (item[2].x, item[2].y)
            bp2 = (item[3].x, item[3].y)
            bp3 = (item[4].x, item[4].y)
            bpts = bezier_to_points(bp0, bp1, bp2, bp3)
            if not current:
                current.extend(bpts)
            else:
                current.extend(bpts[1:])
        elif itype == 'm':
            if len(current) >= 2:
                segments.append(current)
            current = [(item[1].x, item[1].y)]
        elif itype == 're':
            r = item[1]
            rect_pts = [(r.x0, r.y0), (r.x1, r.y0),
                        (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
            if len(current) >= 2:
                segments.append(current)
            segments.append(rect_pts)
            current = []

    if len(current) >= 2:
        segments.append(current)

    return segments


def is_border_path(rect, w_page, h_page):
    """
    パスのbounding rectがページの大部分を占める場合 True を返す（外枠判定）。
    """
    if rect is None:
        return False
    rw = rect[2] - rect[0]
    rh = rect[3] - rect[1]
    return rw > w_page * BORDER_RECT_FRAC and rh > h_page * BORDER_RECT_FRAC


def is_border_seg(pts, w_page, h_page):
    """
    長くてページ端近くに位置する軸平行直線セグメントを外枠と判定する。
    対象: 水平線がページ上下端近く、垂直線がページ左右端近く、かつ十分に長い。
    """
    if len(pts) < 2:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = xmax - xmin
    span_y = ymax - ymin

    # 大型矩形（外枠全体が1セグメントになっているケース）
    if span_x > w_page * BORDER_RECT_FRAC and span_y > h_page * BORDER_RECT_FRAC:
        return True

    # 長い水平線 + ページ上下端近く
    total_len = sum(math.sqrt((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2)
                    for i in range(len(pts)-1))
    if span_y < 20 and total_len > w_page * BORDER_LONG_FRAC:
        if ymin < BORDER_EDGE_MARGIN or ymax > h_page - BORDER_EDGE_MARGIN:
            return True

    # 長い垂直線 + ページ左右端近く
    if span_x < 20 and total_len > h_page * BORDER_LONG_FRAC:
        if xmin < BORDER_EDGE_MARGIN or xmax > w_page - BORDER_EDGE_MARGIN:
            return True

    return False


def project_onto_polyline(px, py, pts):
    """
    点(px,py) をポリライン pts の最近傍点に射影する。
    Returns: (dist, proj_x, proj_y, seg_idx) - 最短距離、投影点座標、分割挿入インデックス
    """
    best_d = float('inf')
    best_proj = (px, py)
    best_sidx = 0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        len2 = dx * dx + dy * dy
        if len2 < 1e-10:
            continue
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / len2))
        qx, qy = x0 + t * dx, y0 + t * dy
        d = math.sqrt((px - qx) ** 2 + (py - qy) ** 2)
        if d < best_d:
            best_d = d
            best_proj = (qx, qy)
            best_sidx = i
    return best_d, best_proj[0], best_proj[1], best_sidx


def load_primary_ss_names(csv_tr_path):
    """変圧器実績CSV → 1次変電所名セットを返す"""
    primary_ss_names = set()
    EXCLUDE_WORDS = {'変電所名', '変電所No.', '変電所No', '二次電圧', '一次電圧',
                     '設備容量', '月', '年', '系統', 'No', 'No.'}
    EXCLUDE_PAT = re.compile(r'^[\d\s※＊\*]+$|^.{1}$')
    jp_pattern  = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff]')
    try:
        df_tr_raw = pd.read_csv(csv_tr_path, encoding='cp932', header=None, low_memory=False)
        print(f"  変圧器CSV: {df_tr_raw.shape[0]}行 x {df_tr_raw.shape[1]}列")
        for ri in [4, 3, 5]:
            if ri >= len(df_tr_raw):
                continue
            row = df_tr_raw.iloc[ri].tolist()
            names = []
            for cell in row:
                s = str(cell).strip()
                if (jp_pattern.search(s)
                        and s not in EXCLUDE_WORDS
                        and not EXCLUDE_PAT.match(s)
                        and 'kV' not in s and 'MW' not in s
                        and '電圧' not in s and '電力' not in s
                        and '変電所' not in s and 'No' not in s):
                    names.append(s)
            if len(names) >= 2:
                print(f"  変電所名行 (行{ri}): {names}")
                primary_ss_names.update(names)
                break
        print(f"  1次変電所リスト ({len(primary_ss_names)}件): {sorted(primary_ss_names)}")
    except Exception as e:
        print(f"  警告: 変圧器CSV読み込み失敗 - {e}")
    return primary_ss_names


def load_line_csv_ids(csv_line_path):
    """線路実績CSV → 線路IDセットを返す"""
    line_csv_ids = set()
    try:
        df_line_raw = pd.read_csv(csv_line_path, encoding='cp932', header=None)
        for ri in range(min(6, len(df_line_raw))):
            row = df_line_raw.iloc[ri].tolist()
            count = sum(1 for c in row if re.match(r'^[0-9]{3,5}[A-Za-z]?$', str(c).strip()))
            if count >= 3:
                print(f"  線路名ヘッダ行: {ri}行目 (線路番号候補 {count}件)")
                for cell in row:
                    s = str(cell).strip()
                    if re.match(r'^[0-9]{3,5}[A-Za-z]?$', s):
                        line_csv_ids.add(s)
                print(f"  線路実績CSV: {len(line_csv_ids)}本の線路名を取得")
                break
        else:
            print("  警告: 線路名ヘッダ行が見つかりませんでした")
    except Exception as e:
        print(f"  警告: 線路CSV読み込み失敗 - {e}")
    return line_csv_ids


def get_leaves(cc, gb):
    return [(n, gb.nodes[n]['x'], gb.nodes[n]['y'])
            for n in cc if gb.degree(n) == 1]


# ─────────────────────────────────────────────
# メインパイプライン
# ─────────────────────────────────────────────

def run_pipeline(region_id: str):
    if region_id not in REGIONS:
        print(f"ERROR: unknown region_id '{region_id}'. Valid: {list(REGIONS.keys())}")
        return

    region      = REGIONS[region_id]
    region_name = region['name']
    prefix      = region['prefix']

    # 地域別線幅パラメータ（REGIONS辞書から取得、なければデフォルト値）
    w66_min, w66_max = region.get('w66', (WIDTH_66KV_MIN, WIDTH_66KV_MAX))
    w33_min, w33_max = region.get('w33', (WIDTH_33KV_MIN, WIDTH_33KV_MAX))

    # ── パス組み立て ── ファイルはすべて INPUT_DIR 直下
    pdf_path   = os.path.join(INPUT_DIR, f"jisseki_{prefix}_map_2024_02.pdf")
    csv_tr     = os.path.join(INPUT_DIR, f"jisseki_{prefix}_tr_2024_02.csv")
    csv_line   = os.path.join(INPUT_DIR, f"jisseki_{prefix}_line_2024_02.csv")
    out_dir    = os.path.join(OUTPUT_BASE, region_id)
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("=" * 70)
    print(f"パイプライン開始: {region_id} {region_name}  (prefix={prefix})")
    print(f"  PDF : {pdf_path}")
    print(f"  OUT : {out_dir}")
    print("=" * 70)

    if not os.path.exists(pdf_path):
        print(f"  WARNING: PDF not found → {pdf_path}")
        print("  スキップします。")
        return

    # ─────────────────────────────────────────────
    # 0. CSV読み込み
    # ─────────────────────────────────────────────
    print()
    print("0. CSVデータ読み込み")
    primary_ss_names = load_primary_ss_names(csv_tr)
    line_csv_ids     = load_line_csv_ids(csv_line)

    # ─────────────────────────────────────────────
    # A. 図形抽出
    # ─────────────────────────────────────────────
    print()
    print("A. 図形抽出 (PDF → 線分集合 + 設備記号)")

    doc  = fitz.open(pdf_path)
    page = doc[PAGE_NUM]

    # ページサイズ: mediabox基準（常に portrait 842×1191）
    W_PAGE = page.mediabox.width
    H_PAGE = page.mediabox.height

    # 回転対応: rotation=90 のページではテキスト座標を mediabox 空間へ変換
    rotation = page.rotation
    has_rotation = (rotation != 0)
    if has_rotation:
        de_mat = page.derotation_matrix
        print(f"  ページ回転検出: rotation={rotation}, derotation_matrix={de_mat}")
    else:
        de_mat = None

    paths = page.get_drawings()
    print(f"  ページサイズ: {W_PAGE:.0f}×{H_PAGE:.0f}pt  (mediabox)")
    print(f"  総パス数: {len(paths)}")

    raw_segs       = []
    seg_id         = 0
    gen_symbols    = []
    circle_symbols = []

    # 凡例エリア除外境界 (mediabox座標)
    LEGEND_X_MIN = 100
    LEGEND_Y_MAX = H_PAGE - 91   # 1100pt相当を比率で算出

    border_path_skip = 0
    for path in paths:
        color  = path.get('color')
        fill   = path.get('fill')
        width  = path.get('width')
        dashes = path.get('dashes')
        items  = path.get('items', [])
        rect   = path.get('rect')

        if width is None:
            width = 0.0

        # ── 外枠フィルタ: ページ大部分を覆う矩形パスをスキップ ──
        if is_border_path(rect, W_PAGE, H_PAGE):
            border_path_skip += 1
            continue

        # ── 発電所シンボル検出（水色fill） ──
        if fill is not None and is_gen_color(fill):
            if rect is not None:
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w >= 3 and h >= 3:
                    cx = (rect[0] + rect[2]) / 2
                    cy = (rect[1] + rect[3]) / 2
                    if cx > LEGEND_X_MIN and cy < LEGEND_Y_MAX:
                        gen_symbols.append({'x': cx, 'y': cy, 'w': w, 'h': h,
                                            'fill': 'blue', 'rect': list(rect)})
            continue

        # ── 発電所シンボル検出（白塗り・黒枠矩形） ──
        if (fill == (1.0, 1.0, 1.0)
                and color is not None and all(c < 0.2 for c in color)
                and 0.5 < width < 1.5
                and rect is not None):
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            if 8 <= w <= 17 and 4 <= h <= 8 and w / h >= 1.4:
                cx = (rect[0] + rect[2]) / 2
                cy = (rect[1] + rect[3]) / 2
                if cx > LEGEND_X_MIN and cy < LEGEND_Y_MAX:
                    gen_symbols.append({'x': cx, 'y': cy, 'w': w, 'h': h,
                                        'fill': 'white', 'rect': list(rect)})
                continue

        # ── 円形シンボル検出（変電所・開閉所） ──
        if rect is not None:
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            if 1 < w <= 25 and 1 < h <= 25:
                aspect_ok = abs(w - h) / max(w, h) < 0.3
                if aspect_ok:
                    cx = (rect[0] + rect[2]) / 2
                    cy = (rect[1] + rect[3]) / 2
                    fill_str  = str(fill)  if fill  is not None else 'none'
                    color_str = str(color) if color is not None else 'none'
                    is_filled_dark  = (fill is not None and is_dark(fill))
                    is_outline_dark = is_dark(color)
                    if is_filled_dark or is_outline_dark:
                        circle_symbols.append({
                            'x': cx, 'y': cy, 'w': w, 'h': h,
                            'fill': fill_str, 'color': color_str,
                            'is_filled': is_filled_dark
                        })

        # ── 線分セグメント変換 ──
        seg_list = path_to_segments(path)
        for pts in seg_list:
            lp = polyline_length(pts)
            raw_segs.append({
                'seg_id': seg_id,
                'page': PAGE_NUM,
                'points': pts,
                'stroke_width': width,
                'stroke_color': color,
                'dashes': dashes,
                'is_dashed': is_dashed(dashes),
                'fill': fill,
                'length_pdf': lp
            })
            seg_id += 1

    print(f"  生成セグメント数: {len(raw_segs)} (外枠除外パス: {border_path_skip})")
    print(f"  発電所シンボル候補: {len(gen_symbols)}")
    print(f"  円形シンボル候補: {len(circle_symbols)}")

    # segments_raw.csv
    rows_raw = []
    for s in raw_segs:
        p = s['points']
        rows_raw.append({
            'seg_id': s['seg_id'],
            'x0': p[0][0], 'y0': p[0][1],
            'x1': p[-1][0], 'y1': p[-1][1],
            'stroke_width': s['stroke_width'],
            'stroke_color': str(s['stroke_color']),
            'dashes': str(s['dashes']),
            'is_dashed': s['is_dashed'],
            'length_pdf': round(s['length_pdf'], 3),
            'n_pts': len(p)
        })
    df_raw = pd.DataFrame(rows_raw)
    df_raw.to_csv(os.path.join(out_dir, 'segments_raw.csv'), index=False, encoding='utf-8-sig')
    print(f"  → segments_raw.csv: {len(df_raw)} 行")

    # ─────────────────────────────────────────────
    # B. 送電線候補抽出 (66kV / 33kV)
    # ─────────────────────────────────────────────
    print()
    print("B. 送電線候補抽出 (66kV / 33kV)")

    segs_66kv = []
    segs_33kv = []
    border_seg_skip = 0

    for s in raw_segs:
        w      = s['stroke_width']
        c      = s['stroke_color']
        l      = s['length_pdf']
        dashed = s['is_dashed']

        ok_color   = is_dark(c)
        ok_len     = l >= SEG_LEN_MIN
        ok_no_fill = s['fill'] is None

        if ok_color and ok_len and ok_no_fill:
            # 外枠・枠線セグメントを除外（長くてページ端寄りの軸平行線）
            if is_border_seg(s['points'], W_PAGE, H_PAGE):
                border_seg_skip += 1
                continue
            # ベジェ曲線由来の多点セグメントを除外（県境ライン等）
            if len(s['points']) > N_PTS_SEG_MAX:
                border_seg_skip += 1
                continue
            if w66_min <= w <= w66_max:
                if not dashed or l < 50.0:
                    segs_66kv.append(s)
            elif w33_min <= w <= w33_max:
                segs_33kv.append(s)

    print(f"  66kV線候補: {len(segs_66kv)} 本")
    print(f"  33kV線候補: {len(segs_33kv)} 本")
    print(f"  外枠除外セグメント: {border_seg_skip} 本")

    for s in segs_66kv:
        s['voltage_kv'] = 66
    for s in segs_33kv:
        s['voltage_kv'] = 33

    filtered_segs = segs_66kv + segs_33kv
    print(f"  合計: {len(filtered_segs)} 本")

    rows_filt = []
    for s in filtered_segs:
        p = s['points']
        rows_filt.append({
            'seg_id':       s['seg_id'],
            'x0': p[0][0], 'y0': p[0][1],
            'x1': p[-1][0], 'y1': p[-1][1],
            'stroke_width': s['stroke_width'],
            'voltage_kv':   s['voltage_kv'],
            'length_pdf':   round(s['length_pdf'], 3),
            'n_pts':        len(p)
        })
    df_filt = pd.DataFrame(rows_filt)
    df_filt.to_csv(os.path.join(out_dir, 'segments_filtered.csv'), index=False, encoding='utf-8-sig')
    print(f"  → segments_filtered.csv: {len(df_filt)} 行")

    # ─────────────────────────────────────────────
    # B2. シンボル→送電線スナップ (浮いた設備を送電線に吸着)
    # ─────────────────────────────────────────────
    print()
    print("B2. シンボル→送電線スナップ")

    all_symbols_b2 = (
        [{'x': c['x'], 'y': c['y'], 'kind': 'circle'} for c in circle_symbols]
        + [{'x': g['x'], 'y': g['y'], 'kind': 'gen'}  for g in gen_symbols]
    )

    # 既存の送電線端点 KDTree
    ep_xy_b2 = (
        [(s['points'][0][0],  s['points'][0][1])  for s in filtered_segs]
        + [(s['points'][-1][0], s['points'][-1][1]) for s in filtered_segs]
    )
    if ep_xy_b2:
        kd_ep_b2 = KDTree(np.array(ep_xy_b2))
    else:
        kd_ep_b2 = None

    snap_added = 0
    fallback_added = 0
    # セグメント追加は後でまとめて filtered_segs に追加
    segs_to_add_b2 = []

    for sym in all_symbols_b2:
        sx, sy = sym['x'], sym['y']

        # 既存端点に十分近い → スナップ不要
        if kd_ep_b2 is not None:
            d_near, _ = kd_ep_b2.query([sx, sy], k=1)
            if d_near <= R_ATTACH * 2.0:
                continue

        # 全 filtered_segs へ投影して最近傍を探す
        best_d    = float('inf')
        best_sidx = -1
        best_px   = sx
        best_py   = sy
        best_ins  = 0

        for si, seg in enumerate(filtered_segs):
            if seg is None:
                continue
            d, qx, qy, ins = project_onto_polyline(sx, sy, seg['points'])
            if d < best_d:
                best_d    = d
                best_sidx = si
                best_px   = qx
                best_py   = qy
                best_ins  = ins

        if best_sidx < 0:
            continue

        if best_d <= SNAP_DIST_LINE:
            # セグメントを投影点で分割 + シンボル→投影点のスタブを追加
            orig = filtered_segs[best_sidx]
            orig_pts = orig['points']

            # 分割後の2セグメント
            pts_a = orig_pts[: best_ins + 1] + [(best_px, best_py)]
            pts_b = [(best_px, best_py)] + orig_pts[best_ins + 1:]
            stub  = [(best_px, best_py), (sx, sy)]

            for spts in [pts_a, pts_b, stub]:
                l_new = polyline_length(spts)
                if len(spts) >= 2 and l_new >= SEG_LEN_MIN:
                    segs_to_add_b2.append({
                        'seg_id':       seg_id,
                        'page':         PAGE_NUM,
                        'points':       spts,
                        'stroke_width': orig['stroke_width'],
                        'stroke_color': orig['stroke_color'],
                        'dashes':       None,
                        'is_dashed':    False,
                        'fill':         None,
                        'length_pdf':   l_new,
                        'voltage_kv':   orig['voltage_kv'],
                    })
                    seg_id += 1

            # 元セグメントを削除 (置換インデックスをあとで処理)
            filtered_segs[best_sidx] = None  # マーク
            snap_added += 1

        elif best_d <= SNAP_FALLBACK:
            # 孤立ノードとして追加 (ステップHのブリッジングで処理)
            orig = filtered_segs[best_sidx]
            stub = [(sx, sy), (sx + 0.1, sy)]  # 長さほぼゼロの自己スタブ
            segs_to_add_b2.append({
                'seg_id':       seg_id,
                'page':         PAGE_NUM,
                'points':       stub,
                'stroke_width': orig['stroke_width'],
                'stroke_color': orig['stroke_color'],
                'dashes':       None,
                'is_dashed':    False,
                'fill':         None,
                'length_pdf':   0.1,
                'voltage_kv':   orig['voltage_kv'],
            })
            seg_id += 1
            fallback_added += 1

    # None マーク済みを削除し、新セグメントを追加
    filtered_segs = [s for s in filtered_segs if s is not None] + segs_to_add_b2
    print(f"  スナップ分割: {snap_added} 箇所, フォールバック追加: {fallback_added} 箇所")
    print(f"  filtered_segs (B2後): {len(filtered_segs)} 本")

    # ─────────────────────────────────────────────
    # C. 接続点同一点化 (DBSCAN スナップ)
    # ─────────────────────────────────────────────
    print()
    print(f"C. 接続点同一点化 (DBSCAN eps={EPS_SNAP} pt)")

    endpoints = []
    for s in filtered_segs:
        p = s['points']
        endpoints.append((p[0][0],  p[0][1],  s['seg_id'], 0))
        endpoints.append((p[-1][0], p[-1][1], s['seg_id'], 1))

    if endpoints:
        ep_arr = np.array([(e[0], e[1]) for e in endpoints])
        db     = DBSCAN(eps=EPS_SNAP, min_samples=1).fit(ep_arr)
        labels = db.labels_
    else:
        labels = []

    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append((endpoints[i][0], endpoints[i][1]))

    junctions = {}
    for jid, pts in clusters.items():
        arr = np.array(pts)
        junctions[jid] = (arr[:, 0].mean(), arr[:, 1].mean())

    print(f"  端点数: {len(endpoints)}, ジャンクション数: {len(junctions)}")

    ep_junction = {}
    for i, lbl in enumerate(labels):
        ep_junction[(endpoints[i][2], endpoints[i][3])] = lbl

    rows_junc = []
    for jid, (cx, cy) in sorted(junctions.items()):
        members = sum(1 for lbl in labels if lbl == jid)
        rows_junc.append({'junction_id': jid, 'x': cx, 'y': cy, 'members': members})
    df_junc = pd.DataFrame(rows_junc)
    df_junc.to_csv(os.path.join(out_dir, 'junctions.csv'), index=False, encoding='utf-8-sig')
    print(f"  → junctions.csv: {len(df_junc)} 行")

    # ─────────────────────────────────────────────
    # D. ネットワーク化
    # ─────────────────────────────────────────────
    print()
    print("D. ネットワーク化")

    G = nx.MultiGraph()
    for jid, (cx, cy) in junctions.items():
        G.add_node(jid, x=cx, y=cy)

    self_loops = 0
    for s in filtered_segs:
        sid = s['seg_id']
        j0  = ep_junction.get((sid, 0))
        j1  = ep_junction.get((sid, 1))
        if j0 is None or j1 is None:
            continue
        if j0 == j1:
            self_loops += 1
            continue

        pts      = s['points']
        geom_str = json.dumps([[round(x, 3), round(y, 3)] for x, y in pts])
        G.add_edge(j0, j1, seg_id=sid, length_pdf=s['length_pdf'],
                   voltage_kv=s['voltage_kv'], geometry=geom_str, line_id=None)

    print(f"  ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
    print(f"  自己ループ除外: {self_loops} 本")
    cc_list = list(nx.connected_components(G))
    print(f"  連結成分数: {len(cc_list)}")
    sizes = sorted([len(c) for c in cc_list], reverse=True)
    print(f"  CC規模 (上位10): {sizes[:10]}")
    deg_cnt = Counter(d for _, d in G.degree())
    print("  次数分布:", dict(sorted(deg_cnt.items())))

    # ─────────────────────────────────────────────
    # E. 路線(line_id)割当 + CSV名称付与
    # ─────────────────────────────────────────────
    print()
    print("E. 路線(line_id)割当")

    # テキスト抽出 (rotation対応)
    blocks = page.get_text("dict")["blocks"]
    line_labels = []

    for b in blocks:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                t  = span["text"].strip()
                bx_disp = (span['bbox'][0] + span['bbox'][2]) / 2
                by_disp = (span['bbox'][1] + span['bbox'][3]) / 2

                # rotation=90 の場合、テキスト座標を mediabox 空間へ変換
                if has_rotation and de_mat is not None:
                    pt_mb = fitz.Point(bx_disp, by_disp) * de_mat
                    bx, by = pt_mb.x, pt_mb.y
                else:
                    bx, by = bx_disp, by_disp

                # タイトル/凡例領域を除外 (mediabox座標)
                if by < 70 or by > H_PAGE - 50:
                    continue

                if re.match(r'^[0-9]{3,5}[A-Za-z]?$', t):
                    line_labels.append({'label': t, 'x': bx, 'y': by})

    df_labels = pd.DataFrame(line_labels)
    if not df_labels.empty:
        df_labels.to_csv(os.path.join(out_dir, 'line_labels.csv'), index=False, encoding='utf-8-sig')
    print(f"  PDF検出ラベル数: {len(line_labels)}")

    # エッジ中点 KDTree
    edge_keys = []
    edge_mids = []
    for u, v, k, data in G.edges(data=True, keys=True):
        geom    = json.loads(data['geometry'])
        mid_idx = len(geom) // 2
        edge_mids.append(geom[mid_idx])
        edge_keys.append((u, v, k))

    if edge_mids and line_labels:
        kd_edges = KDTree(np.array(edge_mids))
        for ll in line_labels:
            pt = np.array([[ll['x'], ll['y']]])
            dist, idx = kd_edges.query(pt, k=1)
            if dist[0] <= R_LABEL:
                u, v, k = edge_keys[idx[0]]
                if G[u][v][k]['line_id'] is None:
                    G[u][v][k]['line_id'] = ll['label']

    assigned = sum(1 for u, v, k, d in G.edges(data=True, keys=True) if d['line_id'] is not None)
    print(f"  line_id割当済み: {assigned} / {G.number_of_edges()}")

    for u, v, k, data in G.edges(data=True, keys=True):
        lid = data.get('line_id')
        data['in_csv'] = (lid in line_csv_ids) if lid else False

    # 未割当に自動採番
    auto_id = 1
    for u, v, k, data in G.edges(data=True, keys=True):
        if data['line_id'] is None:
            data['line_id'] = f"L{auto_id:04d}"
            auto_id += 1

    # ─────────────────────────────────────────────
    # F. ノード確定 + 変電所種別分類
    # ─────────────────────────────────────────────
    print()
    print("F. ノード確定 + 変電所種別分類")

    circ_arr = np.array([[c['x'], c['y']] for c in circle_symbols]) if circle_symbols else None
    gen_arr  = np.array([[g['x'], g['y']] for g in gen_symbols])    if gen_symbols    else None

    kd_circ = KDTree(circ_arr) if circ_arr is not None else None
    kd_gen  = KDTree(gen_arr)  if gen_arr  is not None else None

    # テキストからノード名を取得 (rotation対応)
    substation_texts = []
    for b in blocks:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                t       = span["text"].strip()
                bx_disp = (span['bbox'][0] + span['bbox'][2]) / 2
                by_disp = (span['bbox'][1] + span['bbox'][3]) / 2

                if has_rotation and de_mat is not None:
                    pt_mb = fitz.Point(bx_disp, by_disp) * de_mat
                    bx, by = pt_mb.x, pt_mb.y
                else:
                    bx, by = bx_disp, by_disp

                # タイトル/凡例領域除外
                if by < 70 or by > H_PAGE - 50:
                    continue

                if len(t) >= 2 and not re.match(r'^[0-9A-Za-z]+$', t):
                    substation_texts.append({'text': t, 'x': bx, 'y': by})

    if substation_texts:
        text_arr = np.array([[t['x'], t['y']] for t in substation_texts])
        kd_text  = KDTree(text_arr)
    else:
        kd_text = None

    rows_buses  = []
    bus_id_map  = {}
    bus_id      = 0
    junc_ids    = list(junctions.keys())

    # 発電所シンボル→ジャンクション 1対1マッチング
    junction_to_gen     = {}
    gen_matched_indices = set()

    if kd_gen is not None and junc_ids:
        junc_arr   = np.array([[junctions[jid][0], junctions[jid][1]] for jid in junc_ids])
        kd_junc_lc = KDTree(junc_arr)
        for gi, g in enumerate(gen_symbols):
            dist_j, idx_j = kd_junc_lc.query([g['x'], g['y']], k=1)
            if dist_j <= GEN_R_ATTACH:
                jid_matched = junc_ids[int(idx_j)]
                if jid_matched not in junction_to_gen:
                    junction_to_gen[jid_matched] = gi
                    gen_matched_indices.add(gi)

    for jid in junc_ids:
        cx, cy = junctions[jid]
        deg    = G.degree(jid)

        symbol_type = 'junction'
        is_filled   = False

        if jid in junction_to_gen:
            symbol_type = 'generator'

        if symbol_type == 'junction' and kd_circ is not None:
            dist_c, idx_c = kd_circ.query([cx, cy], k=1)
            if dist_c <= R_ATTACH:
                symbol_type = 'substation_or_sw'
                is_filled   = circle_symbols[int(idx_c)]['is_filled']

        name = ''
        if kd_text is not None:
            dist_t, idx_t = kd_text.query([cx, cy], k=1)
            if dist_t <= R_LABEL:
                name = substation_texts[int(idx_t)]['text']

        ss_type = ''
        if symbol_type == 'generator':
            ss_type = '発電所'
        elif symbol_type == 'substation_or_sw':
            if name in primary_ss_names:
                ss_type = '1次変電所'
            elif is_filled:
                ss_type = '変電所'
            else:
                ss_type = '開閉所'

        rows_buses.append({
            'bus_id':      bus_id,
            'x':           round(cx, 3),
            'y':           round(cy, 3),
            'degree':      deg,
            'symbol_type': symbol_type,
            'ss_type':     ss_type,
            'name':        name,
            'cross_prefecture': False,
            'border_side': ''
        })
        bus_id_map[jid] = bus_id
        bus_id += 1

    # ジャンクションと紐付かなかった発電所シンボルを追加
    if junc_ids:
        junc_arr_full = np.array([[junctions[jid][0], junctions[jid][1]] for jid in junc_ids])
        kd_junc_full  = KDTree(junc_arr_full)
    else:
        kd_junc_full = None

    standalone_edges = []

    for gi, g in enumerate(gen_symbols):
        if gi not in gen_matched_indices:
            gx, gy = g['x'], g['y']
            connected_target = None
            if kd_junc_full is not None:
                k_search = min(10, len(junc_ids))
                dists_f, idxs_f = kd_junc_full.query([gx, gy], k=k_search)
                iter_d = [dists_f] if k_search == 1 else dists_f
                iter_i = [idxs_f] if k_search == 1 else idxs_f
                for df_, idx_f in zip(iter_d, iter_i):
                    if df_ > GEN_FALLBACK_R:
                        break
                    jid_c = junc_ids[int(idx_f)]
                    if G.degree(jid_c) > 0:
                        connected_target = (jid_c, round(df_, 2))
                        break

            gen_bus_id = bus_id
            rows_buses.append({
                'bus_id':      gen_bus_id,
                'x':           round(gx, 3),
                'y':           round(gy, 3),
                'degree':      1 if connected_target else 0,
                'symbol_type': 'generator',
                'ss_type':     '発電所',
                'name':        '',
                'cross_prefecture': False,
                'border_side': ''
            })
            if connected_target:
                jid_c, dist_c = connected_target
                target_bus_id = bus_id_map.get(jid_c)
                if target_bus_id is not None:
                    standalone_edges.append((gen_bus_id, target_bus_id, dist_c))
            bus_id += 1

    df_buses = pd.DataFrame(rows_buses)
    df_buses.to_csv(os.path.join(out_dir, 'buses.csv'), index=False, encoding='utf-8-sig')
    print(f"  → buses.csv: {len(df_buses)} 行")
    print("  symbol_type分布:", df_buses['symbol_type'].value_counts().to_dict())
    print("  ss_type分布:    ", df_buses['ss_type'].value_counts().to_dict())
    print(f"  発電所: ジャンクション紐付={len(gen_matched_indices)}個, "
          f"独立追加={len(gen_symbols)-len(gen_matched_indices)}個")

    # ─────────────────────────────────────────────
    # G. 線路長算出
    # ─────────────────────────────────────────────
    print()
    print("G. 線路長算出")

    rows_lines = []
    for u, v, k, data in G.edges(data=True, keys=True):
        b0 = bus_id_map.get(u)
        b1 = bus_id_map.get(v)
        if b0 is None or b1 is None:
            continue
        rows_lines.append({
            'line_id':         data['line_id'],
            'bus0_id':         b0,
            'bus1_id':         b1,
            'voltage_kv':      data['voltage_kv'],
            'length_pdf':      round(data['length_pdf'], 3),
            'in_csv':          data.get('in_csv', False),
            'geometry':        data['geometry'],
            'connection_type': 'actual',
        })

    for gen_bid, target_bid, dist_c in standalone_edges:
        auto_lid = f"G{auto_id:04d}"
        auto_id += 1
        rows_lines.append({
            'line_id':         auto_lid,
            'bus0_id':         gen_bid,
            'bus1_id':         target_bid,
            'voltage_kv':      66,
            'length_pdf':      round(dist_c, 3),
            'in_csv':          False,
            'geometry':        '',
            'connection_type': 'provisional',
        })

    if standalone_edges:
        print(f"  standalone発電所仮想接続: {len(standalone_edges)}本追加")

    df_lines = pd.DataFrame(rows_lines)
    df_lines.to_csv(os.path.join(out_dir, 'lines.csv'), index=False, encoding='utf-8-sig')
    print(f"  → lines.csv: {len(df_lines)} 行")
    if not df_lines.empty:
        for v in [66, 33]:
            sub = df_lines[df_lines['voltage_kv'] == v]
            if not sub.empty:
                print(f"  {v}kV: {len(sub)}本, length min={sub['length_pdf'].min():.1f} "
                      f"max={sub['length_pdf'].max():.1f} mean={sub['length_pdf'].mean():.1f} pt")

    # ─────────────────────────────────────────────
    # H. ギャップブリッジング
    # ─────────────────────────────────────────────
    print()
    print("H. ギャップブリッジング")

    _df_buses = pd.read_csv(os.path.join(out_dir, 'buses.csv'), encoding='utf-8-sig')
    _df_lines = pd.read_csv(os.path.join(out_dir, 'lines.csv'), encoding='utf-8-sig')

    Gb = nx.Graph()
    for _, b in _df_buses.iterrows():
        Gb.add_node(int(b['bus_id']), x=b['x'], y=b['y'], degree=b['degree'])
    for _, l in _df_lines.iterrows():
        Gb.add_edge(int(l['bus0_id']), int(l['bus1_id']))

    ccs_b = sorted(nx.connected_components(Gb), key=len, reverse=True)
    print(f"  ブリッジ前 CC数: {len(ccs_b)}, 最大CC: {max(len(c) for c in ccs_b)}")

    bridge_rows = []
    bridge_auto = auto_id

    # ── パス1: 通常ブリッジ (≤ BRIDGE_THRESH_PROV) ──
    merged = True
    while merged:
        merged = False
        ccs_b = sorted(nx.connected_components(Gb), key=len, reverse=True)
        if len(ccs_b) <= 1:
            break
        main_leaves = get_leaves(ccs_b[0], Gb)
        for other_cc in ccs_b[1:]:
            other_leaves = get_leaves(other_cc, Gb)
            if not other_leaves:
                other_leaves = [(n, Gb.nodes[n]['x'], Gb.nodes[n]['y']) for n in other_cc]
            best_d, best_pair = float('inf'), None
            for n0, x0, y0 in main_leaves:
                for n1, x1, y1 in other_leaves:
                    d = math.sqrt((x0-x1)**2 + (y0-y1)**2)
                    if d < best_d:
                        best_d, best_pair = d, (n0, n1, x0, y0, x1, y1)
            if best_d <= BRIDGE_THRESH_PROV and best_pair:
                n0, n1, x0, y0, x1, y1 = best_pair
                Gb.add_edge(n0, n1)
                lid   = f"B{bridge_auto:04d}"
                bridge_auto += 1
                ctype = 'bridge' if best_d <= BRIDGE_THRESH_REAL else 'provisional'
                bridge_rows.append({
                    'line_id': lid, 'bus0_id': n0, 'bus1_id': n1,
                    'voltage_kv': 66, 'length_pdf': round(best_d, 3),
                    'in_csv': False, 'geometry': '',
                    'connection_type': ctype
                })
                print(f"  [{ctype}] {lid} [{n0}]({x0:.0f},{y0:.0f}) -- "
                      f"[{n1}]({x1:.0f},{y1:.0f}) dist={best_d:.1f}pt")
                merged = True
                break

    ccs_b = sorted(nx.connected_components(Gb), key=len, reverse=True)
    print(f"  通常ブリッジ後 CC数: {len(ccs_b)}, 最大CC: {max(len(c) for c in ccs_b)}")

    # ── パス2: インセットブリッジ (INSET_BRIDGE_R) ──
    print(f"  インセットブリッジング (R={INSET_BRIDGE_R}pt)")
    inset_count = 0
    merged = True
    while merged:
        merged = False
        ccs_b = sorted(nx.connected_components(Gb), key=len, reverse=True)
        if len(ccs_b) <= 1:
            break
        main_leaves = get_leaves(ccs_b[0], Gb)
        for other_cc in ccs_b[1:]:
            other_leaves = get_leaves(other_cc, Gb)
            if not other_leaves:
                other_leaves = [(n, Gb.nodes[n]['x'], Gb.nodes[n]['y']) for n in other_cc]
            best_d, best_pair = float('inf'), None
            for n0, x0, y0 in main_leaves:
                for n1, x1, y1 in other_leaves:
                    d = math.sqrt((x0-x1)**2 + (y0-y1)**2)
                    if d < best_d:
                        best_d, best_pair = d, (n0, n1, x0, y0, x1, y1)
            if best_d <= INSET_BRIDGE_R and best_pair:
                n0, n1, x0, y0, x1, y1 = best_pair
                Gb.add_edge(n0, n1)
                lid = f"I{bridge_auto:04d}"
                bridge_auto += 1
                bridge_rows.append({
                    'line_id': lid, 'bus0_id': n0, 'bus1_id': n1,
                    'voltage_kv': 66, 'length_pdf': round(best_d, 3),
                    'in_csv': False, 'geometry': '',
                    'connection_type': 'inset_provisional'
                })
                print(f"  [inset_provisional] {lid} [{n0}]({x0:.0f},{y0:.0f}) -- "
                      f"[{n1}]({x1:.0f},{y1:.0f}) dist={best_d:.1f}pt")
                inset_count += 1
                merged = True
                break

    ccs_b = sorted(nx.connected_components(Gb), key=len, reverse=True)
    print(f"  インセットブリッジ後 CC数: {len(ccs_b)}, 最大CC: {max(len(c) for c in ccs_b)}")

    if bridge_rows:
        df_bridge      = pd.DataFrame(bridge_rows)
        _df_lines_new  = pd.concat([_df_lines, df_bridge], ignore_index=True)
        _df_lines_new.to_csv(os.path.join(out_dir, 'lines.csv'), index=False, encoding='utf-8-sig')
        print(f"  → lines.csv 更新: {len(_df_lines_new)} 行")

    # ─────────────────────────────────────────────
    # I. 都道府県境界端点検出
    # ─────────────────────────────────────────────
    print()
    print("I. 都道府県境界端点検出")

    border_count = 0
    bus_rows_updated = _df_buses.copy()
    if 'cross_prefecture' not in bus_rows_updated.columns:
        bus_rows_updated['cross_prefecture'] = False
    if 'border_side' not in bus_rows_updated.columns:
        bus_rows_updated['border_side'] = ''

    # 地図上の実データ範囲を取得
    x_vals = bus_rows_updated['x'].values
    y_vals = bus_rows_updated['y'].values
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_span = x_max - x_min
    y_span = y_max - y_min

    for idx, row in bus_rows_updated.iterrows():
        bid = int(row['bus_id'])
        # degree=1 (端点) かつ symbol_type='junction'（設備なし接続点）
        if Gb.degree(bid) != 1:
            continue
        if str(row.get('symbol_type', '')) != 'junction':
            continue
        x, y = row['x'], row['y']

        # 方向判定: 地図データ範囲の外側25%に近い端点を県境候補とする
        sides = []
        if x < x_min + x_span * 0.20:
            sides.append('W')
        if x > x_max - x_span * 0.20:
            sides.append('E')
        if y < y_min + y_span * 0.20:
            sides.append('N')
        if y > y_max - y_span * 0.20:
            sides.append('S')
        # どの方向でも境界端点として記録（方向不明=empty）
        bus_rows_updated.at[idx, 'cross_prefecture'] = True
        bus_rows_updated.at[idx, 'border_side']      = '/'.join(sides) if sides else 'unknown'
        border_count += 1

    bus_rows_updated.to_csv(os.path.join(out_dir, 'buses.csv'), index=False, encoding='utf-8-sig')
    print(f"  都道府県境界端点: {border_count} ノード")

    # ─────────────────────────────────────────────
    # J. 可視化
    # ─────────────────────────────────────────────
    print()
    print("J. 可視化")

    # 最新CSVを再読み込み
    _df_buses_viz = pd.read_csv(os.path.join(out_dir, 'buses.csv'), encoding='utf-8-sig')
    _df_lines_viz = pd.read_csv(os.path.join(out_dir, 'lines.csv'), encoding='utf-8-sig')

    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, W_PAGE)
    ax.set_ylim(H_PAGE, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    ax.set_xlabel('PDF x (pt)')
    ax.set_ylabel('PDF y (pt)')
    ax.set_title(f"系統図 {region_name} (ID:{region_id})", fontsize=11)

    # ── エッジ描画 ──
    for _, lrow in _df_lines_viz.iterrows():
        ctype = lrow.get('connection_type', 'actual')
        geom  = lrow.get('geometry', '')
        v_kv  = lrow.get('voltage_kv', 66)

        if ctype in ('bridge', 'provisional', 'inset_provisional') or not geom:
            # 座標をバスから取得
            b0_row = _df_buses_viz[_df_buses_viz['bus_id'] == lrow['bus0_id']]
            b1_row = _df_buses_viz[_df_buses_viz['bus_id'] == lrow['bus1_id']]
            if b0_row.empty or b1_row.empty:
                continue
            xs = [b0_row.iloc[0]['x'], b1_row.iloc[0]['x']]
            ys = [b0_row.iloc[0]['y'], b1_row.iloc[0]['y']]
        else:
            try:
                pts = json.loads(geom)
                xs  = [p[0] for p in pts]
                ys  = [p[1] for p in pts]
            except Exception:
                continue

        if ctype == 'actual':
            color = 'navy' if v_kv == 66 else 'purple'
            lw    = 1.4    if v_kv == 66 else 0.8
            ls    = '-'
        elif ctype == 'bridge':
            color = 'gray'
            lw    = 0.8
            ls    = '--'
        elif ctype == 'provisional':
            color = 'gray'
            lw    = 0.6
            ls    = '--'
        elif ctype == 'inset_provisional':
            color = 'darkorange'
            lw    = 0.7
            ls    = ':'
        else:
            color = 'lightgray'
            lw    = 0.5
            ls    = '-'

        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2, alpha=0.85)

    # ── バス描画 ──
    for _, brow in _df_buses_viz.iterrows():
        bx, by = brow['x'], brow['y']
        stype  = brow.get('symbol_type', 'junction')
        sstype = brow.get('ss_type', '')
        name   = brow.get('name', '')
        cross  = brow.get('cross_prefecture', False)

        if stype == 'generator':
            ax.scatter(bx, by, s=50, c='cyan', marker='^', zorder=6)
        elif stype == 'substation_or_sw':
            if sstype == '1次変電所':
                ax.scatter(bx, by, s=60, c='red', marker='s', zorder=7)
            elif sstype == '変電所':
                ax.scatter(bx, by, s=35, c='blue', marker='o', zorder=6)
                ax.scatter(bx, by, s=35, facecolors='none', edgecolors='blue', marker='o', zorder=6)
            else:  # 開閉所
                ax.scatter(bx, by, s=25, facecolors='none', edgecolors='blue',
                           marker='o', linewidths=0.8, zorder=5)
        else:
            ax.scatter(bx, by, s=8, c='darkgreen', marker='.', zorder=4)

        # 変電所名ラベル
        if name and stype == 'substation_or_sw':
            ax.text(bx, by - 3, name, fontsize=4, ha='center', va='bottom', zorder=8)

        # 都道府県境界端点マーカー
        if cross:
            ax.scatter(bx, by, s=80, c='magenta', marker='x', linewidths=1.5, zorder=9)

    # ── 凡例 ──
    leg_handles = [
        mpatches.Patch(color='navy',        label='66kV'),
        mpatches.Patch(color='purple',      label='33kV'),
        mpatches.Patch(color='gray',        label='bridge/provisional (dashed)'),
        mpatches.Patch(color='darkorange',  label='inset_provisional (dotted)'),
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='red',    markersize=7, label='1次変電所'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',   markersize=6, label='変電所'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='blue', markersize=6, label='開閉所'),
        plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='cyan',   markersize=6, label='発電所'),
        plt.Line2D([0],[0], marker='x', color='magenta', markersize=7,       label='都道府県境界端点', lw=0),
    ]
    ax.legend(handles=leg_handles, fontsize=7, loc='upper right')

    out_img = os.path.join(out_dir, f"network_{region_id}.png")
    plt.tight_layout()
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out_img}")

    # ─────────────────────────────────────────────
    # 完了サマリ
    # ─────────────────────────────────────────────
    print()
    print(f"完了サマリ [{region_id} {region_name}]")
    print(f"  segments_raw:      {len(df_raw)} 行")
    print(f"  segments_filtered: {len(df_filt)} 行  (66kV: {len(segs_66kv)}, 33kV: {len(segs_33kv)})")
    print(f"  junctions:         {len(df_junc)} 行")
    print(f"  buses:             {len(_df_buses_viz)} 行")
    print(f"  lines (最終):       {len(_df_lines_viz)} 行")
    print(f"  bridge追加:         {len(bridge_rows)} 本  (うちinset: {inset_count}本)")
    print(f"  CC数 (最終):        {len(ccs_b)}, 最大CC: {max(len(c) for c in ccs_b)}")
    print(f"  都道府県境界端点:    {border_count} ノード")
    print(f"  出力先:             {out_dir}")
    print()


# ─────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if arg == 'all':
        for rid in REGIONS:
            run_pipeline(rid)
    else:
        run_pipeline(arg)


if __name__ == '__main__':
    main()
