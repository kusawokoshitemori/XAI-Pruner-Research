# 診断ログ（diag）

`ログ出したいもの.txt` の「層A：通常の観測ログ」と「第0段階：整合性ログ」を実装したもの。
**手法の挙動は変えない**（乱数を引かない・tensor を in-place 変更しない・scores / candidates / masks を書き換えない）。
`--diag_dir` を付けたときだけ有効になり、付けなければ従来どおり動く。

確認済み: 小さい ViT（DeiT 構造）と ResNet-18 で、元コード / 変更後コード（ログなし）/ 変更後コード（ログあり）の
スコア・枝刈り後の重み・`state.yaml`・実行後の NumPy / Python 乱数状態が完全に一致した。

## 使い方

```bash
python prune-ViT.py ...（従来の引数）... --diag_dir ./diag_logs
python tools/summarize_diag.py ./diag_logs/<run_id>/rank0     # summary.md を生成
```

| 引数 | 既定 | 意味 |
|---|---|---|
| `--diag_dir` | 空（無効） | 出力先 |
| `--diag_lrp_batches` | 2 | LRP 内部（加算・Linear・Filter・Conv・BN）の詳細ログを取る先頭バッチ数。重いので少数に |
| `--diag_snapshots` | 5 | 修復前に NaN/Inf が出た演算の tensor を `snapshots/*.pt` に保存する最大回数 |
| `--diag_near_zero` | 1e-4 | 近ゼロ分母の診断閾値（誤計算の確定ラベルではない） |
| `--diag_flip_threshold` | 1e-8 | 符号反転の閾値付き集計で使う下限 |

スコア集計・EA のログは全バッチ・全世代で取る（軽い）。

## 出力ファイル（`<diag_dir>/<run_id>/rank<r>/`）

| ファイル | 内容 | 対応する疑問 |
|---|---|---|
| `metadata.json` | 引数・Python/torch 版・git commit・seed（Python `random` は未設定であること）など | 0段階, 11.11 |
| `events.jsonl` | 可変長イベント（下表） | 各種 |
| `initial_relevance.csv` | サンプルごとの正解・予測クラス・確信度・初期関連度量（画像ハッシュで識別） | 11.9 |
| `score_batches.csv` / `score_accumulation.csv` | バッチごとのスコアと momentum 累積の統計 | 11.8 |
| `scores_momentum.npz` / `scores_plain_mean.npz` / `scores_after_protect.npz` | 最終スコア、診断用の単純平均、protect 後 | 11.8, 8.4 |
| `protect.csv` | protect 前後の統計、要求保護数と実保護数 | 8.4, 11.6 |
| `component_scores.csv` | 構造ごとの S, n, 非ゼロ要素数, g, S/n, S/g（ViT） | S1, S2 |
| `random_generator.csv` | ランダム補充の初期値・反復数・終了理由・合法性 | 11.4, 11.5 |
| `ea_attempts.csv` | 交叉・突然変異の **1試行1行**（親 ID、交叉マスク、変異判定・変異量、revise 前後、親重複、kids 既存、新規候補、新規マスク） | E1, E2, E6, 10.3, 10.4, 11.1–11.3 |
| `ea_operator_calls.csv` | 演算子呼出ごとの停止理由・一意候補数・一意マスク数 | E3 |
| `candidate_evaluations.csv` | 世代ごとにプール全候補の削除率・削除数・Fitness 内訳・制約量・合法性・順位・選択有無・生成源 | P1 の準備, 8.5, E4 |

### events.jsonl の主なイベント

| event | 内容 | 疑問 |
|---|---|---|
| `filter_runtime_config` | 全 Filter の実際の keep 値（`--filter_percent` が反映されているか） | 0段階, 2.4 |
| `residual_add` | 加算点の shortcut / functional の L2・RMS・比・cos・相殺、分母（raw / used / 近ゼロ）、修復前後の分岐別関連度、q_y_abs、近ゼロ分母由来の割合 | G1, G2 |
| `gate` | Gate が受け取った関連度と返した関連度（カスタム backward 内で記録） | G3 |
| `filter_rescale` | Filtering 前 / マスク後 / 再スケール後、scale とその分母、負 scale・0 分母、選択数と非ゼロ数、選別範囲と正規化範囲、近ゼロ分母と除去の対応表 | F1, F3, F4 |
| `linear_lrp` | 分母、データ項（修復前後）、バイアス項、合計（修復前後）、符号反転、相殺 C、保存の相対差、0 入力への非 0 関連度 | B1, B2, F2, 7.6 |
| `conv2d_lrp` | Conv の分母・修復前非有限・保存 | F2, 7.6 |
| `batchnorm_config` | BN が固定統計量を使っているか（`training` だけで判断しない） | G4 |
| `score_aggregation_comparison` | momentum 集計と単純平均の Spearman・下位集合一致率 | 11.8 |
| `protect_check` / `score_version` | 保護数 0 のとき全 Head が Inf になる問題の検出 / mask_id の基準となるスコア版 | 11.6 |
| `ea_generation` | 世代ごとの選択割合（個体・マスク）、重複マスク、生成源別の生存率と構成比（候補単位・マスク単位）、tie、非合法候補数 | E4, E5, E6 |
| `final_consistency` | 最良候補の削除数 ↔ 最終マスクの削除数、最終マスクから再計算した Fitness、ResNet の共有チャネル調整前後 | I1, 11.7, 12 |
| `model_build` | 実際のパラメータ削減率・`get_complexity` による FLOPs 削減率と目標 | I1 |
| `timing` | ログ処理時間と手法時間（推定）の分離 | 15.4 |

## ID について

- `candidate_id`: 削除率ベクトルを丸めずにハッシュ化したもの
- `mask_id`: スコア順位は protect 後に固定なので、`(score_version, 種類別削除数)` でマスクが一意に決まる。EA ではこれを使う
- `final_mask_hash`: 最終マスク本体のハッシュ

## このログでは扱っていないもの（手法を変える・追加計算が必要なため別実験）

- 層B（FP32/FP64 再計算、再分配あり/なし等）: `snapshots/*.pt` に保存した演算を入力に別スクリプトで行う
- 層C（Gating 開閉、Filtering 保持率比較、バイアス規則の比較）
- 層D（候補ごとの削除モデル評価、A・B・AB 削除、部品交換）: `candidate_evaluations.csv` の `rates`/`counts` を入力に使える
- E3 の予算曲線（max_iters を超えて試行を続ける）: 乱数消費が変わるので元の EA と混ぜない
- 局所増幅率 G_i（[出力×入力] の項が大きすぎるため）

## 既存コードの注意点（ログで検出できるが、挙動を変えないため修正していない）

- `prune-CNN.py` は `prune_one_shot()` に `num_classes` を渡しておらず、そのままでは TypeError になる
- `random.sample(self.candidates, k)` は set を渡しているため Python 3.11 以降で TypeError（`metadata.json` に記録）
- CNN の `--filter_percent` は使われておらず Conv の Filter は 0.5 固定（`filter_runtime_config` で確認できる）
