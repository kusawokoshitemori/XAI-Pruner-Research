## 実行例（Base / Small / Tiny, ログ付き）

元の XAIPruner Code では Base モデルを使う前提の実行例が基本です。Small / Tiny は、`--model` と `--resume` だけを差し替えれば同じ流れで実行できます。

```bash
# Base（元の実行例を踏襲）
python prune-ViT.py \
  --model deit_base_patch16_224 \
  --resume ./checkpoint/deit_base_patch16_224.pth \
  --data_path ./data_dl/imagenet \
  --output_dir ./output/base \
  --batch-size 32 \
  --pruning_rate 0.5 \
  --protect_rate 0.25 \
  --population 100 \
  --generation 100 \
  --pruning_momentum 0.9 \
  --diag_dir ./diag_logs
```

```bash
# Small
python prune-ViT.py \
  --model deit_small_patch16_224 \
  --resume ./checkpoint/deit_small_patch16_224.pth \
  --data_path ./data_dl/imagenet \
  --output_dir ./output/small \
  --batch-size 32 \
  --pruning_rate 0.5 \
  --protect_rate 0.25 \
  --population 100 \
  --generation 100 \
  --pruning_momentum 0.9 \
  --diag_dir ./diag_logs
```

```bash
# Tiny
python prune-ViT.py \
  --model deit_tiny_patch16_224 \
  --resume ./checkpoint/deit_tiny_patch16_224.pth \
  --data_path ./data_dl/imagenet \
  --output_dir ./output/tiny \
  --batch-size 32 \
  --pruning_rate 0.5 \
  --protect_rate 0.25 \
  --population 100 \
  --generation 100 \
  --pruning_momentum 0.9 \
  --diag_dir ./diag_logs
```

ログの集計は次を実行します。

```bash
python tools/summarize_diag.py ./diag_logs/<run_id>/rank0
```

## 診断ログ（`--diag_dir`）

`lrp/`, `pruner/`, `engine/`, `prune-ViT.py`, `prune-CNN.py` に、挙動を変えずに内部状態を観測するための診断ログ機能（`diag/`）を追加した。
**`--diag_dir` を指定しない限り（既定）、計算経路・乱数の消費順序・最終的な出力（スコア・マスク・枝刈り後の重み）は元のコードと完全に同じになるよう実装している。** 無効時の各ログ呼び出しは最初の分岐（`if STATE.enabled` 相当）で即座に戻るだけで、テンソルの書き換えや乱数の消費は行わない。有効時も、ログは `detach()` したコピーだけを読んで記録するだけで、関連度・スコア・候補・マスクといった手法側の値を一切書き換えない（方針は `diag/core.py` 冒頭のコメント、検証結果は `diag/README.md` を参照）。

### ログが出るタイミング

| タイミング | 内容 | 既定の頻度 |
|---|---|---|
| 関連度計算（`compute_scores`）の各バッチ | 初期関連度・BatchNorm設定・LRP内部（残差加算・Linear・Filter・Conv）の詳細 | LRP内部の詳細ログのみ先頭 `--diag_lrp_batches`（既定2）バッチに限定。それ以外は全バッチ |
| 全バッチ終了後 | momentum集計 vs 単純平均の比較 | 1回 |
| 進化計算開始前（`protect()` 前後） | 保護前後のスコア統計、構造種類別スコア | 1回 |
| 進化計算の各世代 | ランダム補充・交叉・突然変異の試行ログ、選択結果 | 全世代 |
| 枝刈り確定後 | 候補→マスク→実モデルの整合性、実際のパラメータ/FLOPs削減率 | 1回 |
| ファインチューニング前の評価後（`prune-ViT.py` のみ） | 評価指標 | 1回 |

コード上の起点は `prune-ViT.py` / `prune-CNN.py` の `main()` 内の `diag.init(args, ...)`（`--diag_dir` が空なら何もせず `False` を返す）、バッチ単位の区切りは `engine/engine.py` の `compute_scores()` 内の `diag.begin_batch()` / `diag.end_batch()`、世代単位の区切りは `pruner/pruner.py` の `Controller_vit.engine()` / `Controller_resnet.engine()` 内。

### ログの読み方

生ログ（`<diag_dir>/<run_id>/rank<r>/events.jsonl` と各種 `.csv`）は、フィールド名自体に意味・単位を埋め込み（例: `near_zero_threshold`, `functional_abs_share_q_y_abs`）、解釈を誤りやすい指標には `note_*` 系のフィールドで注記を添えている。それでも値だけを並べた表は読みにくいため、次の集計ツールで人が読める形にまとめてから確認することを推奨する。

```bash
python tools/summarize_diag.py <diag_dir>/<run_id>/rank0
```

`summary.md` が生成され、ログ追加の動機になった疑問一覧（G1〜I1）に対応する表としてまとまる。出力ファイル・イベントの一覧やIDの定義など詳細は `diag/README.md` を参照。
