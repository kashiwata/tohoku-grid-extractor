# データスキーマ / Data Schema

## buses.csv

変電所・発電所・開閉所・接続点（ノード）の情報。

| カラム | 型 | 説明 |
|--------|----|------|
| `bus_id` | int | バスID（地域内で一意） |
| `x` | float | PDF座標 x [pt] |
| `y` | float | PDF座標 y [pt]（下方向が正） |
| `degree` | int | グラフ次数（接続線路数） |
| `symbol_type` | str | `circle_symbol` / `generator` / `segment_endpoint` |
| `ss_type` | str | 変電所種別（下記参照） |
| `name` | str | 変電所名（テキスト抽出、空欄あり） |
| `cross_prefecture` | bool | 県境接続候補フラグ |
| `border_side` | str | 境界側情報 |

**ss_type の値:**

| 値 | 意味 |
|----|------|
| `1次変電所` | 基幹変電所（154kV/66kV変圧、電源側） |
| `変電所` | 配電用変電所（66kV/6.6kV変圧、負荷側） |
| `開閉所` | 開閉所 |
| `発電所` | 発電所 |
| `""` (空欄) | 送電線の接続点・中間ノード |

---

## lines.csv

送電線・暫定接続（エッジ）の情報。

| カラム | 型 | 説明 |
|--------|----|------|
| `line_id` | str | 線路ID（例: `L0001`） |
| `bus0_id` | int | 始端バスID |
| `bus1_id` | int | 終端バスID |
| `voltage_kv` | int | 電圧階級 [kV]（66 or 33） |
| `length_pdf` | float | PDF上の線路長 [pt] |
| `in_csv` | bool | 系統図に線番が記載されているか |
| `geometry` | str | JSON形式の座標列 `[[x1,y1],[x2,y2],...]` |
| `connection_type` | str | `actual` / `provisional` / `inset_provisional` |

**connection_type の値:**

| 値 | 意味 |
|----|------|
| `actual` | 実際に抽出された送電線 |
| `provisional` | 接続点補完（ギャップブリッジング） |
| `inset_provisional` | 挿入図エリアと本図の接続（暫定） |

---

## power_flow/buses_pf.csv

buses.csv に潮流計算用の電力設定を追加したデータ。

buses.csv の全カラムに加えて以下を追加:

| カラム | 型 | 説明 |
|--------|----|------|
| `bus_type` | str | `SL` / `PV` / `PQ`（下記参照） |
| `load_mw` | float | 負荷有効電力 [MW] |
| `load_mvar` | float | 負荷無効電力 [MVAR] |
| `p_gen_mw` | float | 発電有効電力 [MW] |
| `q_gen_mvar` | float | 発電無効電力 [MVAR] |
| `v_pu` | float | 電圧設定値 [pu]（SL/PV母線のみ有効） |

**bus_type の値:**

| 値 | 意味 | 対象ノード |
|----|------|-----------|
| `SL` | スラック母線（基準母線） | degree最大の1次変電所に接続する上位系統電源（各地域1件） |
| `PV` | 電圧制御母線 | その他1次変電所の上位系統電源（V=1.05pu, P=16MW） |
| `PQ` | 負荷母線 | 変電所（P=4MW）・発電所（P=-5MW）・開閉所・接続点 |

**電力設定の根拠:**

| ノード種別 | 設定値 | 根拠 |
|-----------|--------|------|
| 配電用変電所 | P_load = 4MW | 容量10MVA × 負荷率40% |
| 1次変電所電源 | P_gen = 16MW | 容量40MVA × 供給率40% |
| 発電所 | P_gen = 5MW | 仮置き（小規模再エネ想定） |
| 力率 | 0.90 | 統一仮置き値 |

---

## power_flow/lines_pf.csv

lines.csv に電気パラメータを追加したデータ。

lines.csv の全カラムに加えて以下を追加:

| カラム | 型 | 説明 |
|--------|----|------|
| `length_km` | float | 実距離換算線路長 [km] |
| `r_pu` | float | 抵抗 [pu, 100MVA基準] |
| `x_pu` | float | リアクタンス [pu, 100MVA基準] |
| `b_pu` | float | 対地サセプタンス [pu, 100MVA基準]（Pi モデル全体） |
| `rating_mva` | float | 熱容量 [MVA] |
| `connection_type` | str | `transformer` が追加（変圧器エッジ） |
| `is_transformer` | bool | 変圧器エッジフラグ |

**線路パラメータ（ACSR標準値）:**

| 電圧 | r [Ω/km] | x [Ω/km] | b [μS/km] | 基準インピーダンス |
|------|-----------|-----------|-----------|-----------------|
| 66kV | 0.030 | 0.383 | 2.74 | 43.56 Ω |
| 33kV | 0.0567 | 0.390 | 3.00 | 10.89 Ω |

**変圧器パラメータ（154kV/66kV, 仮置き）:**

| パラメータ | 値 |
|-----------|-----|
| r_pu | 0.005 |
| x_pu | 0.100 |
| 定格 | 80 MVA |

---

## combined/power_flow/

全7地域を統合したデータ。bus_id は地域オフセットで一意化。

| 地域 | bus_id オフセット |
|------|----------------|
| 01 青森 | 0 |
| 02 岩手 | 10,000 |
| 03 秋田 | 20,000 |
| 04 宮城 | 30,000 |
| 05 山形 | 40,000 |
| 06 福島 | 50,000 |
| 07 新潟 | 60,000 |

統合モデルのスラック母線は全地域中1件のみ。他の上位系統電源はPVに降格。
