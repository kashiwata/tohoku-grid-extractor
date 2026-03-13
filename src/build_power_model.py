# -*- coding: utf-8 -*-
"""
build_power_model.py  (v2 – multi-region)
==========================================
全7地域（01-07）の buses.csv / lines.csv から潮流計算用モデルを構築する。

処理フロー:
  A. 地域別処理 (run_region)
     1. buses.csv / lines.csv 読み込み（最大CCのみ使用）
     2. 変電所・発電所に負荷・発電電力を設定
        - 変電所: P_load = 容量10MVA × 40% = 4MW
        - 発電所: P_gen  = 5MW (仮置き)
     3. 1次変電所ごとに上位系統電源ノードを追加
        - degree最大の1次変電所の電源ノード → SL (スラック母線, V=1.05∠0°)
        - その他電源ノード → PV (V=1.05, P=16MW = 40MVA×40%)
     4. 線路電気パラメータ計算 (Pi モデル)
     5. Y-bus 構築
     6. CSV / npy 出力 + SLD 生成
  B. 統合処理 (run_combined)
     - 全地域のデータを統合 (bus_id offset で一意化)
     - 県境跨ぎ接続を自動検出・追加 (lat/lon 近傍 ≤ 0.6°)
     - 統合 Y-bus を出力

出力先:
  output_multi/{rid}/power_flow/    ← 地域別
  output_multi/combined/power_flow/ ← 統合
"""

import os
import math
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# ベースパラメータ
# ============================================================
S_BASE_MVA   = 100.0
V_BASE_66_KV = 66.0
V_BASE_33_KV = 33.0
Z_BASE_66    = V_BASE_66_KV**2 / S_BASE_MVA   # 43.56 Ω
Z_BASE_33    = V_BASE_33_KV**2 / S_BASE_MVA   # 10.89 Ω

# ============================================================
# 地域パラメータ
# ============================================================
REGION_PARAMS = {
    '01': {'name': '青森県', 'pt_to_km': 0.17},
    '02': {'name': '岩手県', 'pt_to_km': 0.22},
    '03': {'name': '秋田県', 'pt_to_km': 0.19},
    '04': {'name': '宮城県', 'pt_to_km': 0.10},
    '05': {'name': '山形県', 'pt_to_km': 0.14},
    '06': {'name': '福島県', 'pt_to_km': 0.14},
    '07': {'name': '新潟県', 'pt_to_km': 0.23},
}

# 統合モデル用 bus_id オフセット（地域間で一意になるよう 10000 刻み）
REGION_OFFSET = {
    '01':     0,
    '02': 10000,
    '03': 20000,
    '04': 30000,
    '05': 40000,
    '06': 50000,
    '07': 60000,
}

# ============================================================
# 電力設定
# ============================================================
CAP_MVA = {
    '1次変電所': 40.0,
    '変電所':    10.0,
    '開閉所':     0.0,
}
LOAD_RATE    = 0.40   # 配電用変電所 負荷率
SRC_RATE     = 0.40   # 1次変電所 供給率
GEN_MW_UNIT  = 5.0    # 発電所1基あたり出力 [MW]
POWER_FACTOR = 0.90   # 統一力率

# ============================================================
# 送電線パラメータ（ACSR標準値）
# ============================================================
LINE_PARAMS = {
    66: {
        'r_per_km':   0.030,
        'x_per_km':   0.383,
        'b_per_km':   2.74e-6,   # [S/km]
        'rating_mva': 50.0,
    },
    33: {
        'r_per_km':   0.0567,
        'x_per_km':   0.390,
        'b_per_km':   3.00e-6,
        'rating_mva': 20.0,
    },
}

# ============================================================
# 変圧器パラメータ（154kV/66kV）
# ============================================================
XFRM_X_PU       = 0.10
XFRM_R_PU       = 0.005
XFRM_RATING_MVA = 80.0

# 県境跨ぎ接続
CROSS_PREF_KM      = 50.0   # デフォルト線路長 [km]
CROSS_PREF_DEG     = 0.6    # 近傍判定距離 [度] ≈ 55km

# ============================================================
# パス
# ============================================================
BASE_DIR = r"C:\Users\taku.kashiwazaki\Documents\PowerSystem"


def region_dir(rid):
    return os.path.join(BASE_DIR, f"output_multi/{rid}")


def power_flow_dir(rid):
    d = os.path.join(BASE_DIR, f"output_multi/{rid}/power_flow")
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================
# 1. データ読み込み + 最大CC抽出
# ============================================================
def load_region(rid):
    buses_csv = os.path.join(region_dir(rid), "buses.csv")
    lines_csv = os.path.join(region_dir(rid), "lines.csv")

    if not os.path.exists(buses_csv) or not os.path.exists(lines_csv):
        print(f"  [SKIP] {rid}: CSVなし ({buses_csv})")
        return None, None

    buses = pd.read_csv(buses_csv)
    lines = pd.read_csv(lines_csv)
    buses.columns = [c.lstrip('\ufeff') for c in buses.columns]
    lines.columns = [c.lstrip('\ufeff') for c in lines.columns]
    buses['bus_id']  = buses['bus_id'].astype(int)
    lines['bus0_id'] = lines['bus0_id'].astype(int)
    lines['bus1_id'] = lines['bus1_id'].astype(int)

    # 最大CCを抽出
    G = nx.Graph()
    for _, b in buses.iterrows():
        G.add_node(int(b['bus_id']))
    for _, l in lines.iterrows():
        G.add_edge(int(l['bus0_id']), int(l['bus1_id']))

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    main_nodes = components[0]
    print(f"  [{rid}] CC数={len(components)}, 最大CC={len(main_nodes)}ノード")

    buses_mc = buses[buses['bus_id'].isin(main_nodes)].copy().reset_index(drop=True)
    mask_l   = lines['bus0_id'].isin(main_nodes) & lines['bus1_id'].isin(main_nodes)
    lines_mc = lines[mask_l].copy().reset_index(drop=True)

    return buses_mc, lines_mc


# ============================================================
# 2. バス電力設定
# ============================================================
def assign_bus_power(buses_mc):
    """
    各バスに bus_type='PQ', load_mw, load_mvar, p_gen_mw, q_gen_mvar, v_pu を付与。
    電源ノード（上位系統電源）は add_source_nodes で設定するため除外。
    """
    tan_phi = math.tan(math.acos(POWER_FACTOR))
    rows = []

    for _, row in buses_mc.iterrows():
        st  = str(row.get('ss_type', ''))
        sym = str(row.get('symbol_type', ''))

        load_mw = 0.0
        gen_mw  = 0.0

        if st == '変電所':
            load_mw = CAP_MVA.get('変電所', 10.0) * LOAD_RATE   # 4MW
        elif sym == 'generator' or st == '発電所':
            gen_mw = GEN_MW_UNIT                                 # 5MW

        load_mvar = load_mw * tan_phi if load_mw > 0 else 0.0
        gen_mvar  = gen_mw  * tan_phi if gen_mw  > 0 else 0.0

        rows.append({
            'bus_type':   'PQ',
            'load_mw':    round(load_mw,   4),
            'load_mvar':  round(load_mvar, 4),
            'p_gen_mw':   round(gen_mw,    4),
            'q_gen_mvar': round(gen_mvar,  4),
            'v_pu':       1.0,
        })

    extra = pd.DataFrame(rows, index=buses_mc.index)
    return pd.concat([buses_mc, extra], axis=1)


# ============================================================
# 3. 上位系統電源ノード追加 (SL / PV)
# ============================================================
def add_source_nodes(buses_mc, lines_mc):
    """
    1次変電所ごとに上位系統電源ノードを追加。
    - degree最大の1次変電所 → SL (スラック, V=1.05∠0°)
    - その他 → PV (V=1.05, P=16MW)
    変圧器エッジも新規追加して返す。
    """
    tan_phi = math.tan(math.acos(POWER_FACTOR))
    p_src   = CAP_MVA.get('1次変電所', 40.0) * SRC_RATE   # 16MW

    primary_subs = buses_mc[buses_mc['ss_type'] == '1次変電所'].copy()

    # 名前ごとに代表ノード (degree最大) を1つ選ぶ
    rep_nodes = {}
    for name, grp in primary_subs.groupby('name'):
        if not name or (isinstance(name, float) and np.isnan(name)):
            continue
        rep_row = grp.loc[grp['degree'].idxmax()]
        rep_nodes[name] = rep_row

    # 名前なし1次変電所は bus_id をキーに
    for _, row in primary_subs.iterrows():
        key = str(row.get('name', ''))
        if not key or key in ('nan', ''):
            rep_nodes[f"sub_{row['bus_id']}"] = row

    # スラック母線: degree最大の1次変電所
    if rep_nodes:
        slack_name = max(rep_nodes.keys(),
                         key=lambda k: int(rep_nodes[k].get('degree', 0)))
    else:
        slack_name = None

    next_bid = int(buses_mc['bus_id'].max()) + 1
    next_lid = 9000
    new_buses = []
    new_lines = []

    for name, rep in rep_nodes.items():
        src_id   = next_bid; next_bid += 1
        is_slack = (name == slack_name)
        btype    = 'SL' if is_slack else 'PV'
        p_inj    = 0.0  if is_slack else p_src
        q_inj    = 0.0  if is_slack else p_src * tan_phi

        # lat/lon 継承（あれば）
        src_row = {
            'bus_id':      src_id,
            'x':           float(rep['x']),
            'y':           float(rep['y']) - 40.0,
            'degree':      1,
            'symbol_type': 'source',
            'ss_type':     '上位系統電源',
            'name':        f"{name}_SRC",
            'bus_type':    btype,
            'load_mw':     0.0,
            'load_mvar':   0.0,
            'p_gen_mw':    round(p_inj, 4),
            'q_gen_mvar':  round(q_inj, 4),
            'v_pu':        1.05,
        }
        for col in ('lat', 'lon', 'cross_prefecture'):
            if col in rep.index:
                src_row[col] = rep[col]
        new_buses.append(src_row)

        lid = f"XFRM{next_lid:04d}"; next_lid += 1
        new_lines.append({
            'line_id':         lid,
            'bus0_id':         src_id,
            'bus1_id':         int(rep['bus_id']),
            'voltage_kv':      66,
            'length_pdf':      0.0,
            'length_km':       0.0,
            'r_pu':            XFRM_R_PU,
            'x_pu':            XFRM_X_PU,
            'b_pu':            0.0,
            'rating_mva':      XFRM_RATING_MVA,
            'connection_type': 'transformer',
            'is_transformer':  True,
        })

    if new_buses:
        src_df    = pd.DataFrame(new_buses)
        buses_out = pd.concat([buses_mc, src_df], ignore_index=True)
    else:
        buses_out = buses_mc.copy()

    return buses_out, lines_mc, new_lines


# ============================================================
# 4. 線路電気パラメータ計算
# ============================================================
def compute_line_params(lines_mc, pt_to_km):
    """
    lines_mc の各行に length_km, r_pu, x_pu, b_pu, rating_mva を付与。
    変圧器エッジはスキップ（既に設定済み）。
    """
    rows = []
    for _, row in lines_mc.iterrows():
        if row.get('is_transformer', False):
            rows.append(row.to_dict())
            continue

        v_kv  = int(row.get('voltage_kv', 66))
        param = LINE_PARAMS.get(v_kv, LINE_PARAMS[66])
        z_base = Z_BASE_66 if v_kv == 66 else Z_BASE_33

        l_pt = float(row.get('length_pdf', 0.0))
        l_km = l_pt * pt_to_km

        r_pu = param['r_per_km'] * l_km / z_base
        x_pu = param['x_per_km'] * l_km / z_base
        b_pu = param['b_per_km'] * l_km * z_base  # b [S] / Y_base = b * Z_base

        d = row.to_dict()
        d['length_km']  = round(l_km, 4)
        d['r_pu']       = round(r_pu, 6)
        d['x_pu']       = round(x_pu, 6)
        d['b_pu']       = round(b_pu, 8)
        d['rating_mva'] = param['rating_mva']
        rows.append(d)

    return pd.DataFrame(rows)


# ============================================================
# 5. Y-bus アドミタンス行列の構築
# ============================================================
def build_ybus(buses_pf, lines_pf_all):
    """
    複素アドミタンス行列 Y [n×n] を Pi モデルで構築する。
      対角: Σ(y_ij + jb_ij/2)
      非対角: -y_ij
    """
    bid_list   = list(buses_pf['bus_id'].astype(int))
    bid_to_idx = {b: i for i, b in enumerate(bid_list)}
    n = len(bid_list)
    Y = np.zeros((n, n), dtype=complex)

    for _, row in lines_pf_all.iterrows():
        u = int(row['bus0_id'])
        v = int(row['bus1_id'])
        if u not in bid_to_idx or v not in bid_to_idx:
            continue
        i = bid_to_idx[u]
        j = bid_to_idx[v]

        r = float(row.get('r_pu', 0.0))
        x = float(row.get('x_pu', 1e-6))
        b = float(row.get('b_pu', 0.0))

        z        = complex(r, x)
        y_series = 1.0 / z if abs(z) > 1e-12 else complex(0, 1e6)
        y_shunt  = complex(0, b / 2.0)

        Y[i, j] -= y_series
        Y[j, i] -= y_series
        Y[i, i] += y_series + y_shunt
        Y[j, j] += y_series + y_shunt

    return Y, bid_to_idx


# ============================================================
# 6. 出力 (CSV / npy / SLD)
# ============================================================
def save_outputs(buses_pf, lines_pf_all, Y, bid_to_idx, out_dir, title):
    buses_pf.to_csv(f"{out_dir}/buses_pf.csv",    index=False, encoding='utf-8-sig')
    lines_pf_all.to_csv(f"{out_dir}/lines_pf.csv", index=False, encoding='utf-8-sig')

    idx_list  = [b for b, _ in sorted(bid_to_idx.items(), key=lambda kv: kv[1])]
    ybus_real = pd.DataFrame(Y.real, index=idx_list, columns=idx_list)
    ybus_imag = pd.DataFrame(Y.imag, index=idx_list, columns=idx_list)
    ybus_real.to_csv(f"{out_dir}/ybus_real.csv", encoding='utf-8-sig')
    ybus_imag.to_csv(f"{out_dir}/ybus_imag.csv", encoding='utf-8-sig')
    np.save(f"{out_dir}/ybus_complex.npy", Y)

    n       = len(buses_pf)
    nonzero = np.count_nonzero(Y)
    diag_g  = np.diag(Y.real)
    diag_b  = np.diag(Y.imag)
    print(f"  Y-bus: {n}×{n}, 非ゼロ={nonzero}")
    print(f"  対角(G): min={diag_g.min():.4f} max={diag_g.max():.4f} mean={diag_g.mean():.4f}")
    print(f"  対角(B): min={diag_b.min():.4f} max={diag_b.max():.4f} mean={diag_b.mean():.4f}")

    _draw_sld(buses_pf, lines_pf_all, f"{out_dir}/network_sld.png", title)
    print(f"  → {out_dir}/")


def _draw_sld(buses_pf, lines_pf_all, out_path, title):
    fig, ax = plt.subplots(figsize=(18, 24))
    H_PAGE  = 1191

    # 線描画
    for _, row in lines_pf_all.iterrows():
        u, v   = int(row['bus0_id']), int(row['bus1_id'])
        b0_ser = buses_pf[buses_pf['bus_id'] == u]
        b1_ser = buses_pf[buses_pf['bus_id'] == v]
        if b0_ser.empty or b1_ser.empty:
            continue
        b0, b1 = b0_ser.iloc[0], b1_ser.iloc[0]

        ctype  = str(row.get('connection_type', 'actual'))
        is_xfm = bool(row.get('is_transformer', False))
        v_kv   = int(row.get('voltage_kv', 66))
        geom   = row.get('geometry', '')

        if not geom or (isinstance(geom, float) and np.isnan(geom)):
            xs = [float(b0['x']), float(b1['x'])]
            ys = [float(b0['y']), float(b1['y'])]
        else:
            try:
                pts = json.loads(str(geom))
                xs  = [p[0] for p in pts]
                ys  = [p[1] for p in pts]
            except Exception:
                xs = [float(b0['x']), float(b1['x'])]
                ys = [float(b0['y']), float(b1['y'])]

        if is_xfm:
            ax.plot(xs, ys, color='darkorange', lw=2.5, zorder=3)
        elif ctype == 'provisional':
            ax.plot(xs, ys, color='#AAAAAA', lw=0.8, zorder=1, linestyle=(0, (4, 3)))
        else:
            color = '#1A1A8C' if v_kv == 66 else '#7C3AED'
            ax.plot(xs, ys, color=color, lw=1.4 if v_kv == 66 else 0.8, zorder=2)

    # ノード描画
    for _, row in buses_pf.iterrows():
        px, py = float(row['x']), float(row['y'])
        st     = str(row.get('ss_type', ''))
        sym    = str(row.get('symbol_type', ''))
        name   = str(row.get('name', ''))
        lmw    = float(row.get('load_mw', 0.0))
        gmw    = float(row.get('p_gen_mw', 0.0))

        if lmw > 0:
            ax.annotate('', xy=(px, py + 12), xytext=(px, py + 4),
                        arrowprops=dict(arrowstyle='->', color='#CC4400', lw=1.0), zorder=5)
        if gmw > 0:
            ax.annotate('', xy=(px, py - 12), xytext=(px, py - 4),
                        arrowprops=dict(arrowstyle='->', color='#006600', lw=1.0), zorder=5)

        if st == '上位系統電源':
            btype = str(row.get('bus_type', ''))
            ax.scatter(px, py, marker='^', s=120, color='darkorange',
                       edgecolors='black', linewidths=0.8, zorder=6)
            ax.text(px + 4, py - 6, name.replace('_SRC', f'\n({btype})'),
                    fontsize=5.5, color='darkorange', va='top', zorder=7)
        elif sym == 'generator' or st == '発電所':
            ax.scatter(px, py, marker='^', s=50, color='cyan',
                       edgecolors='navy', linewidths=0.5, zorder=6)
        elif st == '1次変電所':
            ax.scatter(px, py, marker='s', s=100, color='red',
                       edgecolors='darkred', linewidths=0.8, zorder=6)
            ax.text(px + 5, py, name, fontsize=7, color='darkred',
                    va='center', fontweight='bold', zorder=7)
        elif st == '変電所':
            ax.scatter(px, py, marker='o', s=60, color='royalblue',
                       edgecolors='navy', linewidths=0.8, zorder=6)
            if name and name not in ('nan', ''):
                ax.text(px + 4, py, name, fontsize=5.5, color='navy', va='center', zorder=7)
        elif st == '開閉所':
            ax.scatter(px, py, marker='o', s=50, facecolors='white',
                       edgecolors='royalblue', linewidths=1.0, zorder=6)
            if name and name not in ('nan', ''):
                ax.text(px + 4, py, name, fontsize=5.0, color='steelblue', va='center', zorder=7)
        else:
            ax.scatter(px, py, marker='D', s=10, color='#888888',
                       edgecolors='none', zorder=5)

    ax.set_xlim(0, 842)
    ax.set_ylim(H_PAGE, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('#F8F8F8')
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("x [pt]")
    ax.set_ylabel("y [pt]")

    legend_elements = [
        Line2D([0], [0], color='darkorange', lw=2.5, label='変圧器 (154/66kV)'),
        Line2D([0], [0], color='#1A1A8C',   lw=1.4, label='66kV 送電線'),
        Line2D([0], [0], color='#7C3AED',   lw=0.8, label='33kV 送電線'),
        Line2D([0], [0], color='#AAAAAA',   lw=0.8, linestyle='--', label='暫定接続'),
        mpatches.Patch(color='darkorange', label='上位系統電源 (SL/PV)'),
        mpatches.Patch(color='red',        label='1次変電所'),
        mpatches.Patch(color='royalblue',  label='配電用変電所'),
        mpatches.Patch(color='white',      label='開閉所',
                       edgecolor='royalblue', linewidth=1),
        mpatches.Patch(color='cyan',       label='発電所'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
              framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  SLD保存: {out_path}")


# ============================================================
# 7. 県境跨ぎ接続の検出
# ============================================================
def find_cross_pref_connections(all_data):
    """
    all_data: {rid: {'buses_pf': df, ...}} のdict（bus_idはオフセット前の元値）
    lat/lon カラムを使って異地域間の近傍ノードペアを接続候補として返す。
    bus_id にはオフセット済み値を使用。
    """
    cross_nodes = []

    for rid, dat in all_data.items():
        buses  = dat['buses_pf']
        offset = REGION_OFFSET[rid]

        if 'lat' not in buses.columns or 'lon' not in buses.columns:
            continue

        if 'cross_prefecture' in buses.columns:
            mask = buses['cross_prefecture'].astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            # degree=1 かつ外縁10%以内
            if len(buses) < 4:
                continue
            lat_lo = buses['lat'].quantile(0.10)
            lat_hi = buses['lat'].quantile(0.90)
            lon_lo = buses['lon'].quantile(0.10)
            lon_hi = buses['lon'].quantile(0.90)
            mask = (buses['degree'] == 1) & (
                (buses['lat'] < lat_lo) | (buses['lat'] > lat_hi) |
                (buses['lon'] < lon_lo) | (buses['lon'] > lon_hi)
            )

        for _, row in buses[mask].iterrows():
            cross_nodes.append({
                'rid':    rid,
                'bus_id': int(row['bus_id']) + offset,
                'lat':    float(row['lat']),
                'lon':    float(row['lon']),
                'name':   str(row.get('name', '')),
            })

    if len(cross_nodes) < 2:
        print(f"  県境跨ぎ候補ノード不足 ({len(cross_nodes)}件) — スキップ")
        return []

    # 異地域ペアで距離判定
    connections = []
    used_pairs  = set()

    for i in range(len(cross_nodes)):
        for j in range(i + 1, len(cross_nodes)):
            ni, nj = cross_nodes[i], cross_nodes[j]
            if ni['rid'] == nj['rid']:
                continue
            pair_key = (min(ni['bus_id'], nj['bus_id']),
                        max(ni['bus_id'], nj['bus_id']))
            if pair_key in used_pairs:
                continue

            dlat     = ni['lat'] - nj['lat']
            dlon     = ni['lon'] - nj['lon']
            dist_deg = math.sqrt(dlat**2 + dlon**2)

            if dist_deg <= CROSS_PREF_DEG:
                used_pairs.add(pair_key)
                connections.append({
                    'line_id':         f"XPREF_{ni['bus_id']}_{nj['bus_id']}",
                    'bus0_id':         ni['bus_id'],
                    'bus1_id':         nj['bus_id'],
                    'voltage_kv':      66,
                    'length_pdf':      0.0,
                    'length_km':       CROSS_PREF_KM,
                    'r_pu':            round(LINE_PARAMS[66]['r_per_km']
                                            * CROSS_PREF_KM / Z_BASE_66, 6),
                    'x_pu':            round(LINE_PARAMS[66]['x_per_km']
                                            * CROSS_PREF_KM / Z_BASE_66, 6),
                    'b_pu':            round(LINE_PARAMS[66]['b_per_km']
                                            * CROSS_PREF_KM * Z_BASE_66, 8),
                    'rating_mva':      50.0,
                    'connection_type': 'cross_prefecture',
                    'is_transformer':  False,
                    'from_rid':        ni['rid'],
                    'to_rid':          nj['rid'],
                })

    print(f"  県境跨ぎ接続: {len(connections)} 本")
    return connections


# ============================================================
# 地域別実行
# ============================================================
def run_region(rid):
    rp = REGION_PARAMS.get(rid)
    if rp is None:
        print(f"  [SKIP] 未知地域: {rid}")
        return None

    name     = rp['name']
    pt_to_km = rp['pt_to_km']
    print(f"\n=== [{rid}] {name} ===")

    # 1. 読み込み
    buses_mc, lines_mc = load_region(rid)
    if buses_mc is None:
        return None
    print(f"  buses={len(buses_mc)}, lines={len(lines_mc)}")

    # 2. バス電力設定
    buses_mc = assign_bus_power(buses_mc)

    # 3. 電源ノード追加
    buses_pf, lines_mc, new_xfm_rows = add_source_nodes(buses_mc, lines_mc)
    print(f"  電源ノード追加後: buses={len(buses_pf)}, 変圧器={len(new_xfm_rows)}本")

    # 4. 線路パラメータ計算
    lines_pf = compute_line_params(lines_mc, pt_to_km)
    if new_xfm_rows:
        xfm_df       = pd.DataFrame(new_xfm_rows)
        lines_pf_all = pd.concat([lines_pf, xfm_df], ignore_index=True)
    else:
        lines_pf_all = lines_pf.copy()

    # 5. Y-bus
    Y, bid_to_idx = build_ybus(buses_pf, lines_pf_all)

    # 6. 出力
    out_dir = power_flow_dir(rid)
    save_outputs(buses_pf, lines_pf_all, Y, bid_to_idx, out_dir,
                 f"{name} 66kV系統単線結線図（仮モデル）")

    # サマリ
    sl_count = (buses_pf.get('bus_type', pd.Series(dtype=str)) == 'SL').sum()
    pv_count = (buses_pf.get('bus_type', pd.Series(dtype=str)) == 'PV').sum()
    pq_count = (buses_pf.get('bus_type', pd.Series(dtype=str)) == 'PQ').sum()
    print(f"  バス種別: SL={sl_count}, PV={pv_count}, PQ={pq_count}")

    non_xfm = lines_pf[~lines_pf.get('is_transformer',
                                      pd.Series([False] * len(lines_pf))
                                      ).fillna(False)]
    v_stat = non_xfm['voltage_kv'].value_counts().to_dict() if 'voltage_kv' in non_xfm.columns else {}
    print(f"  電圧別線路数: {v_stat}")
    if 'length_km' in non_xfm.columns:
        print(f"  総線路長: {non_xfm['length_km'].sum():.1f} km")

    return {
        'rid':        rid,
        'name':       name,
        'buses_pf':   buses_pf,
        'lines_pf':   lines_pf_all,
        'Y':          Y,
        'bid_to_idx': bid_to_idx,
    }


# ============================================================
# 統合モデル実行
# ============================================================
def run_combined(all_data):
    print("\n=== 統合モデル (全地域) ===")

    if not all_data:
        print("  データなし")
        return

    combined_buses_list = []
    combined_lines_list = []

    for rid, dat in all_data.items():
        offset  = REGION_OFFSET[rid]
        buses_r = dat['buses_pf'].copy()
        lines_r = dat['lines_pf'].copy()

        buses_r['bus_id']  = buses_r['bus_id'].astype(int) + offset
        buses_r['region']  = rid
        lines_r['bus0_id'] = lines_r['bus0_id'].astype(int) + offset
        lines_r['bus1_id'] = lines_r['bus1_id'].astype(int) + offset
        lines_r['region']  = rid

        combined_buses_list.append(buses_r)
        combined_lines_list.append(lines_r)

    combined_buses = pd.concat(combined_buses_list, ignore_index=True)
    combined_lines = pd.concat(combined_lines_list, ignore_index=True)

    # 統合モデルでは SL は1つだけ。残りを PV に降格
    sl_mask = combined_buses['bus_type'] == 'SL'
    sl_idxs = combined_buses[sl_mask].index.tolist()
    if len(sl_idxs) > 1:
        for idx in sl_idxs[1:]:
            combined_buses.at[idx, 'bus_type'] = 'PV'
        print(f"  統合SL調整: SL=1, PVに降格={len(sl_idxs)-1}件")

    # 県境跨ぎ接続（オフセット前の buses_pf を使って検出）
    cross_connections = find_cross_pref_connections(all_data)
    if cross_connections:
        cross_df       = pd.DataFrame(cross_connections)
        combined_lines = pd.concat([combined_lines, cross_df], ignore_index=True)

    print(f"  統合: buses={len(combined_buses)}, lines={len(combined_lines)}")

    # Y-bus
    Y_comb, bid_to_idx_comb = build_ybus(combined_buses, combined_lines)

    # 出力
    out_dir = os.path.join(BASE_DIR, "output_multi/combined/power_flow")
    os.makedirs(out_dir, exist_ok=True)

    combined_buses.to_csv(f"{out_dir}/buses_pf.csv",    index=False, encoding='utf-8-sig')
    combined_lines.to_csv(f"{out_dir}/lines_pf.csv",    index=False, encoding='utf-8-sig')

    idx_list  = [b for b, _ in sorted(bid_to_idx_comb.items(), key=lambda kv: kv[1])]
    ybus_real = pd.DataFrame(Y_comb.real, index=idx_list, columns=idx_list)
    ybus_imag = pd.DataFrame(Y_comb.imag, index=idx_list, columns=idx_list)
    ybus_real.to_csv(f"{out_dir}/ybus_real.csv", encoding='utf-8-sig')
    ybus_imag.to_csv(f"{out_dir}/ybus_imag.csv", encoding='utf-8-sig')
    np.save(f"{out_dir}/ybus_complex.npy", Y_comb)

    n       = len(combined_buses)
    nonzero = np.count_nonzero(Y_comb)
    print(f"  統合Y-bus: {n}×{n}, 非ゼロ={nonzero}")
    print(f"  → {out_dir}/")

    for btype in ['SL', 'PV', 'PQ']:
        count = (combined_buses.get('bus_type', pd.Series(dtype=str)) == btype).sum()
        print(f"    {btype}: {count}")


# ============================================================
# メイン
# ============================================================
def main():
    print("=== build_power_model.py (v2 multi-region) ===")
    print(f"対象地域: {list(REGION_PARAMS.keys())}")

    all_data = {}
    for rid in sorted(REGION_PARAMS.keys()):
        result = run_region(rid)
        if result is not None:
            all_data[rid] = result

    print(f"\n処理完了地域: {list(all_data.keys())} ({len(all_data)}/{len(REGION_PARAMS)})")

    run_combined(all_data)

    print("\n=== 完了 ===")
    print("  地域別出力: output_multi/{rid}/power_flow/")
    print("  統合出力:   output_multi/combined/power_flow/")


if __name__ == "__main__":
    main()
