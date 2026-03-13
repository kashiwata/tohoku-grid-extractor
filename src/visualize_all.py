# -*- coding: utf-8 -*-
"""
visualize_all.py
================
全7地域の系統ネットワークを地理座標（緯度・経度）に変換し、
東北電力ネットワーク全体の系統図を 1 枚の PNG で出力する。

改訂版 (2026-03):
  - cos(lat) 補正による正確な地図アスペクト比
  - inset_provisional 接続のノードを地理的正位置へ移動
  - 県境端点 (cross_prefecture) 間の仮想県間接続線
  - bounding-box 近似を除去し全地域アフィン変換に統一

出力: output_multi/network_all.png
"""

import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker
import networkx as nx

matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR    = r"C:\Users\taku.kashiwazaki\Documents\PowerSystem"
OUT_BASE    = os.path.join(BASE_DIR, "output_multi")
OUT_PNG     = os.path.join(OUT_BASE, "network_all.png")

# ============================================================
# アンカー点: {region_id: [(pdf_x, pdf_y, lat, lon), ...]}
# 1次変電所の既知緯度経度 (公開地図から推定)
# ============================================================
ANCHORS = {
    '01': [
        # 青森県 (rotation=90 PDF → mediabox座標で指定)
        # buses.csv の変電所名と地理座標を照合した確認済みアンカー
        (666, 731, 40.77, 140.58),   # 石江変電所  (青森市石江)
        (580, 1052, 41.38, 141.22),  # 佐井変電所  (下北郡佐井村)
        (220, 520, 40.50, 141.49),   # 小中野変電所 (八戸市小中野)
        (231, 456, 40.55, 141.60),   # 階上変電所  (三戸郡階上町)
    ],
    '02': [
        # 岩手県
        (367, 484, 39.74, 141.15),   # 好摩変電所  (盛岡市好摩)
        (310, 480, 39.72, 141.10),   # 柏台変電所  (盛岡市)
        (370, 705, 39.44, 141.10),   # 日詰変電所  (紫波町日詰)
        (673, 792, 39.36, 141.89),   # 大槌変電所  (大槌町)
        (227, 833, 38.98, 141.07),   # 後藤野変電所 (一関市)
        (390, 1019, 38.83, 141.43),  # 東山変電所  (一関市東山)
    ],
    '03': [
        # 秋田県
        (278, 520, 39.74, 140.10),   # 土崎変電所  (秋田市土崎)
        (270, 268, 40.18, 140.03),   # 能代東変電所 (能代市)
        (273, 730, 39.38, 140.05),   # 本荘変電所  (由利本荘市)
        (654, 240, 40.18, 140.74),   # 毛馬内変電所 (鹿角市)
        (538, 894, 39.06, 140.48),   # 横堀変電所  (湯沢市)
        (169, 428, 39.89, 139.84),   # 船川変電所  (男鹿市)
        (346, 645, 39.45, 140.30),   # 川添変電所  (大仙市)
        (560, 820, 39.10, 140.51),   # 羽後増田変電所 (横手市)
    ],
    '04': [
        # 宮城県
        (386, 218, 38.71, 141.02),   # 岩ヶ崎変電所 (栗原市)
        (230, 438, 38.53, 140.72),   # 宮崎変電所  (大崎市)
        (402, 690, 38.29, 140.89),   # 大日変電所  (仙台市)
        (549, 550, 38.36, 141.18),   # 北仙台変電所 (仙台市)
        (630, 640, 38.26, 141.25),   # 若林変電所  (仙台市)
        (723, 550, 38.34, 141.50),   # 石巻変電所  (石巻市)
    ],
    '05': [
        # 山形県
        (377, 298, 38.84, 139.74),   # 余目変電所  (庄内町)
        (275, 375, 38.73, 139.87),   # 大山変電所  (鶴岡市)
        (556, 278, 38.96, 140.18),   # 真室川変電所 (真室川町)
        (622, 449, 38.58, 140.38),   # 大石田変電所 (大石田町)
        (577, 603, 38.37, 140.24),   # 寒河江変電所 (寒河江市)
        (451, 917, 38.13, 140.05),   # 小松変電所  (川西町)
        (562, 609, 38.38, 140.28),   # 寒河江2変電所 (寒河江市)
    ],
    '06': [
        # 福島県 (rotation=90 PDF → mediabox座標)
        (157, 521, 36.98, 140.83),   # 湯本第一変電所 (いわき市)
        (316, 694, 37.09, 140.68),   # 小野新町変電所 (小野町)
        (544, 605, 37.13, 140.21),   # 白河変電所  (白河市)
        (420, 430, 37.35, 140.45),   # 郡山変電所  (郡山市)
        (230, 370, 37.35, 140.90),   # 原町変電所  (南相馬市)
        (650, 440, 37.30, 139.95),   # 会津変電所  (会津若松市)
    ],
    '07': [
        # 新潟県 (buses.csv の 1次変電所 確認済みアンカー)
        (101, 1008, 37.04, 137.87),  # 田海変電所   (糸魚川市田海)
        (660, 401,  38.22, 139.49),  # 村上変電所   (村上市)
        (449, 999,  37.04, 138.89),  # 塩沢変電所   (南魚沼市塩沢)
        # buses.csv の変電所名から確認した補助アンカー
        (114, 788,  37.10, 138.22),  # 東直江津変電所 (上越市)
        (471, 783,  37.46, 138.85),  # 北長岡変電所  (長岡市)
        (654, 432,  38.07, 139.49),  # 荒川変電所   (村上市荒川)
        (457, 666,  37.89, 139.03),  # 南新潟変電所  (新潟市南区)
    ],
}

# ============================================================
# アフィン変換
# ============================================================
def compute_affine(anchors):
    n = len(anchors)
    M = np.zeros((n, 3))
    b_lat = np.zeros(n)
    b_lon = np.zeros(n)
    for i, (px, py, lat, lon) in enumerate(anchors):
        M[i] = [px, py, 1.0]
        b_lat[i] = lat
        b_lon[i] = lon
    A_lat, _, _, _ = np.linalg.lstsq(M, b_lat, rcond=None)
    A_lon, _, _, _ = np.linalg.lstsq(M, b_lon, rcond=None)

    # RMSE を計算して表示
    lats_pred = M @ A_lat
    lons_pred = M @ A_lon
    rmse_lat = np.sqrt(np.mean((lats_pred - b_lat)**2))
    rmse_lon = np.sqrt(np.mean((lons_pred - b_lon)**2))
    rmse_km = math.sqrt((rmse_lat * 111)**2 + (rmse_lon * 111 * math.cos(math.radians(b_lat.mean())))**2)
    return A_lat, A_lon, rmse_km


def pdf_to_geo(px, py, A_lat, A_lon):
    lat = A_lat[0]*px + A_lat[1]*py + A_lat[2]
    lon = A_lon[0]*px + A_lon[1]*py + A_lon[2]
    return lat, lon


# ============================================================
# インセット領域の地理座標修正
# ============================================================
def relocate_inset_nodes(bus_geo, buses_df, lines_df):
    """
    inset_provisional で接続されたノードを、
    接続先のメインネットワーク位置の近傍に移動する。

    返り値: 修正後の bus_geo (dict bus_id → (lat, lon))
    """
    if 'connection_type' not in lines_df.columns:
        return bus_geo

    # inset_provisional エッジを特定
    inset_edges = lines_df[lines_df['connection_type'] == 'inset_provisional']
    if inset_edges.empty:
        return bus_geo

    # inset_provisional で繋がれたノードをグラフ化
    G_full = nx.Graph()
    for _, row in lines_df.iterrows():
        G_full.add_edge(int(row['bus0_id']), int(row['bus1_id']),
                        ctype=str(row.get('connection_type', 'actual')))

    # メインCCを特定 (最大CC)
    ccs = sorted(nx.connected_components(G_full), key=len, reverse=True)
    if not ccs:
        return bus_geo
    main_cc = ccs[0]

    # inset_provisional edges の端点を調べる
    # 片方がメインCC、もう片方がサブCC
    inset_sub_to_main = {}  # sub_node → main_node
    for _, row in inset_edges.iterrows():
        b0, b1 = int(row['bus0_id']), int(row['bus1_id'])
        if b0 in main_cc and b1 not in main_cc:
            inset_sub_to_main[b1] = b0
        elif b1 in main_cc and b0 not in main_cc:
            inset_sub_to_main[b0] = b1

    if not inset_sub_to_main:
        return bus_geo

    # サブCCノードをメインノードの位置に移動（平行移動）
    bus_geo_new = dict(bus_geo)  # copy
    processed_ccs = set()

    for sub_entry, main_node in inset_sub_to_main.items():
        # sub_entryが属するCCを取得
        sub_cc = nx.node_connected_component(G_full, sub_entry)
        sub_cc_id = frozenset(sub_cc)
        if sub_cc_id in processed_ccs:
            continue
        processed_ccs.add(sub_cc_id)

        # サブCCの各ノードを取得 (メインCCに属するものを除く)
        sub_nodes = [n for n in sub_cc if n not in main_cc]
        if not sub_nodes:
            continue

        # 接続点のメイン側座標
        if main_node not in bus_geo_new:
            continue
        anchor_lat, anchor_lon = bus_geo_new[main_node]

        # サブCC内のメイン接続端点の現在座標
        if sub_entry not in bus_geo_new:
            continue
        sub_lat, sub_lon = bus_geo_new[sub_entry]

        # 平行移動量
        dlat = anchor_lat - sub_lat
        dlon = anchor_lon - sub_lon

        # サブCC全体を移動
        for n in sub_nodes:
            if n in bus_geo_new:
                lat, lon = bus_geo_new[n]
                bus_geo_new[n] = (lat + dlat, lon + dlon)

    return bus_geo_new


# ============================================================
# 県間接続線の生成
# ============================================================
def build_inter_region_connections(region_data):
    """
    各地域のcross_prefecture=Trueノードから、
    隣接地域の最近傍ノードへの仮想接続を生成する。

    返り値: [(lat0, lon0, lat1, lon1), ...]
    """
    # cross_prefectureノードを全地域から収集
    cross_nodes = []
    for rid, (buses, _, bus_geo, _) in region_data.items():
        for _, row in buses.iterrows():
            if str(row.get('cross_prefecture', '')).lower() in ('true', '1', 'yes'):
                bid = int(row['bus_id'])
                if bid in bus_geo:
                    lat, lon = bus_geo[bid]
                    cross_nodes.append({'rid': rid, 'bid': bid,
                                        'lat': lat, 'lon': lon,
                                        'side': str(row.get('border_side', ''))})

    if len(cross_nodes) < 2:
        return []

    # 異なる地域間の近傍ペアを探す
    connections = []
    used_pairs = set()
    coords = np.array([[n['lat'], n['lon']] for n in cross_nodes])

    for i, ni in enumerate(cross_nodes):
        # 異なる地域のノードのみ
        dists = []
        for j, nj in enumerate(cross_nodes):
            if ni['rid'] == nj['rid']:
                dists.append(float('inf'))
            else:
                dlat = ni['lat'] - nj['lat']
                dlon = (ni['lon'] - nj['lon']) * math.cos(math.radians((ni['lat']+nj['lat'])/2))
                dists.append(math.sqrt(dlat**2 + dlon**2))
        j_best = int(np.argmin(dists))
        if dists[j_best] < 0.6:  # 約55km以内（隣接県境）
            nj = cross_nodes[j_best]
            key = tuple(sorted([(ni['rid'], ni['bid']), (nj['rid'], nj['bid'])]))
            if key not in used_pairs:
                used_pairs.add(key)
                connections.append((ni['lat'], ni['lon'], nj['lat'], nj['lon']))

    return connections


# ============================================================
# メイン処理
# ============================================================
def main():
    print("=== visualize_all.py (revised) ===")

    REGION_COLORS = {
        '01': '#E64C4C',   # 青森 赤
        '02': '#4C8BE6',   # 岩手 青
        '03': '#E6A84C',   # 秋田 橙
        '04': '#9B59B6',   # 宮城 紫
        '05': '#27AE60',   # 山形 緑
        '06': '#C0392B',   # 福島 深赤
        '07': '#1ABC9C',   # 新潟 水緑
    }
    REGION_NAMES = {
        '01': '青森県', '02': '岩手県', '03': '秋田県',
        '04': '宮城県', '05': '山形県', '06': '福島県', '07': '新潟県'
    }

    # 緯度中心値 (Tohoku region)
    LAT_CENTER = 38.5
    # cos(lat) 補正 → 経度 1 度 = cos(lat_center) × 緯度 1 度
    LON_TO_LAT = 1.0 / math.cos(math.radians(LAT_CENTER))  # ≈ 1.286

    fig, ax = plt.subplots(figsize=(26, 30))

    all_lats, all_lons = [], []
    region_data = {}  # rid → (buses, lines, bus_geo, color)

    for region_id in ['01', '02', '03', '04', '05', '06', '07']:
        out_dir = os.path.join(OUT_BASE, region_id)
        buses_path = os.path.join(out_dir, 'buses.csv')
        lines_path = os.path.join(out_dir, 'lines.csv')
        if not os.path.exists(buses_path):
            print(f"  [{region_id}] buses.csv not found, skip")
            continue

        buses = pd.read_csv(buses_path, encoding='utf-8-sig')
        lines = pd.read_csv(lines_path, encoding='utf-8-sig')
        buses.columns = [c.lstrip('\ufeff') for c in buses.columns]
        lines.columns = [c.lstrip('\ufeff') for c in lines.columns]

        color = REGION_COLORS[region_id]
        anchors = ANCHORS.get(region_id, [])
        if len(anchors) < 3:
            print(f"  [{region_id}] アンカー不足 ({len(anchors)}点), スキップ")
            continue

        A_lat, A_lon, rmse_km = compute_affine(anchors)
        print(f"  [{region_id}] {REGION_NAMES[region_id]}: アフィンRMSE={rmse_km:.1f}km")

        # 変換テーブル (bus_id → (lat, lon))
        bus_geo = {}
        for _, row in buses.iterrows():
            bid = int(row['bus_id'])
            lat, lon = pdf_to_geo(float(row['x']), float(row['y']), A_lat, A_lon)
            bus_geo[bid] = (lat, lon)

        # インセット領域を地理的正位置へ移動
        bus_geo = relocate_inset_nodes(bus_geo, buses, lines)

        region_data[region_id] = (buses, lines, bus_geo, color)

        for bid, (lat, lon) in bus_geo.items():
            all_lats.append(lat)
            all_lons.append(lon)

    # ── 県間接続線 ──
    inter_connections = build_inter_region_connections(region_data)
    print(f"  県間仮想接続: {len(inter_connections)} 本")

    for (lat0, lon0, lat1, lon1) in inter_connections:
        ax.plot([lon0, lon1], [lat0, lat1],
                color='black', lw=1.0, ls='--', zorder=3, alpha=0.5)

    # ── 各地域の描画 ──
    for region_id, (buses, lines, bus_geo, color) in region_data.items():
        A_lat, A_lon, _ = compute_affine(ANCHORS[region_id])

        # エッジ描画
        for _, row in lines.iterrows():
            u = int(row['bus0_id'])
            v = int(row['bus1_id'])
            if u not in bus_geo or v not in bus_geo:
                continue

            ctype = str(row.get('connection_type', 'actual'))
            v_kv  = int(row.get('voltage_kv', 66))

            raw_geom = row.get('geometry', '')
            geom_ok = (raw_geom and not (isinstance(raw_geom, float) and np.isnan(raw_geom))
                       and str(raw_geom).strip() not in ('', 'nan'))

            if ctype in ('provisional', 'bridge', 'inset_provisional') or not geom_ok:
                lats = [bus_geo[u][0], bus_geo[v][0]]
                lons = [bus_geo[u][1], bus_geo[v][1]]
            else:
                try:
                    pts = json.loads(str(raw_geom))
                    lats_r = []; lons_r = []
                    for pt in pts:
                        lt, ln = pdf_to_geo(float(pt[0]), float(pt[1]), A_lat, A_lon)
                        lats_r.append(lt); lons_r.append(ln)
                    lats, lons = lats_r, lons_r
                except Exception:
                    lats = [bus_geo[u][0], bus_geo[v][0]]
                    lons = [bus_geo[u][1], bus_geo[v][1]]

            if ctype == 'inset_provisional':
                ax.plot(lons, lats, color='darkorange', lw=0.5, ls=':', zorder=1, alpha=0.5)
            elif ctype in ('provisional', 'bridge'):
                ax.plot(lons, lats, color='#BBBBBB', lw=0.6, ls='--', zorder=1, alpha=0.55)
            else:
                lw = 1.2 if v_kv == 66 else 0.7
                ax.plot(lons, lats, color=color, lw=lw, zorder=2, alpha=0.85,
                        solid_capstyle='round')

        # ノード描画
        for _, row in buses.iterrows():
            bid = int(row['bus_id'])
            if bid not in bus_geo:
                continue
            lat, lon = bus_geo[bid]
            ss   = str(row.get('ss_type', ''))
            sym  = str(row.get('symbol_type', ''))
            name = str(row.get('name', ''))
            cp   = str(row.get('cross_prefecture', '')).lower() in ('true', '1', 'yes')

            if ss == '1次変電所':
                ax.scatter(lon, lat, marker='s', s=90, color=color,
                           edgecolors='black', linewidths=0.9, zorder=7)
                if name and name not in ('nan', ''):
                    ax.text(lon + 0.02, lat + 0.01, name, fontsize=6.5,
                            color='black', fontweight='bold', zorder=8)
            elif sym == 'generator':
                ax.scatter(lon, lat, marker='^', s=22, color='deepskyblue',
                           edgecolors=color, linewidths=0.4, zorder=5, alpha=0.8)
            elif ss == '変電所':
                ax.scatter(lon, lat, marker='o', s=28, color=color,
                           edgecolors='black', linewidths=0.4, zorder=6, alpha=0.9)
                if name and name not in ('nan', ''):
                    ax.text(lon + 0.013, lat, name, fontsize=4,
                            color='black', zorder=7, va='center')
            elif ss == '開閉所':
                ax.scatter(lon, lat, marker='o', s=15, facecolors='white',
                           edgecolors=color, linewidths=0.5, zorder=5, alpha=0.9)
            elif cp:
                ax.scatter(lon, lat, marker='x', s=30, color='magenta',
                           linewidths=1.0, zorder=8, alpha=0.9)
            else:
                ax.scatter(lon, lat, marker='.', s=4, color=color,
                           zorder=4, alpha=0.4)

        # 地域名ラベル
        lats_r = [v[0] for v in bus_geo.values()]
        lons_r = [v[1] for v in bus_geo.values()]
        if lats_r:
            lat_c = np.percentile(lats_r, 50)
            lon_c = np.percentile(lons_r, 50)
            ax.text(lon_c, lat_c, REGION_NAMES[region_id], fontsize=11,
                    color=color, fontweight='bold', ha='center', va='center',
                    zorder=10, alpha=0.5,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='none', alpha=0.6))

        print(f"    buses={len(buses)}, lines={len(lines)}")

    # ── アスペクト比 (cos(lat)補正) ──
    ax.set_aspect(LON_TO_LAT)

    if all_lats:
        lat_pad = (max(all_lats) - min(all_lats)) * 0.04
        lon_pad = (max(all_lons) - min(all_lons)) * 0.04
        ax.set_xlim(min(all_lons) - lon_pad, max(all_lons) + lon_pad)
        ax.set_ylim(min(all_lats) - lat_pad, max(all_lats) + lat_pad)

    ax.set_xlabel('経度 [°E]', fontsize=11)
    ax.set_ylabel('緯度 [°N]', fontsize=11)
    ax.set_title('東北電力ネットワーク 66kV以下系統図（全7県 合成）',
                 fontsize=14, pad=14)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}°E'))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}°N'))

    # ── 凡例 ──
    legend_region = [
        mpatches.Patch(color=REGION_COLORS[rid], label=f"{rid} {REGION_NAMES[rid]}")
        for rid in ['01','02','03','04','05','06','07']
    ]
    legend_type = [
        Line2D([0],[0], color='gray', lw=1.2, label='66kV送電線'),
        Line2D([0],[0], color='gray', lw=0.6, label='33kV送電線'),
        Line2D([0],[0], color='#BBBBBB', lw=0.8, ls='--', label='ブリッジ/仮想接続'),
        Line2D([0],[0], color='darkorange', lw=0.7, ls=':', label='インセット仮接続'),
        Line2D([0],[0], color='black', lw=1.0, ls='--', label='県間仮想接続'),
        plt.scatter([],[], marker='s', s=60, color='gray', label='1次変電所'),
        plt.scatter([],[], marker='o', s=28, color='gray', label='配電用変電所'),
        plt.scatter([],[], marker='o', s=15, facecolors='white',
                    edgecolors='gray', linewidths=0.5, label='開閉所'),
        plt.scatter([],[], marker='^', s=22, color='deepskyblue', label='発電所'),
        plt.scatter([],[], marker='x', s=30, color='magenta', label='県境接続候補'),
    ]
    leg1 = ax.legend(handles=legend_region, loc='lower left',
                     fontsize=8, framealpha=0.9, title='地域', ncol=2)
    ax.add_artist(leg1)
    ax.legend(handles=legend_type, loc='lower right',
              fontsize=7.5, framealpha=0.9, title='凡例')

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  → 出力: {OUT_PNG}")


if __name__ == '__main__':
    main()
