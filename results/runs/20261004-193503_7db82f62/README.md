# 実行 20261004-193503_7db82f62

保存時点: 2026-10-11（Asia/Tokyo）

- 処理: `prune-ViT.py`
- 状態: `pruning_evaluation_completed`
- 条件: `numerical_fixes_and_diagnostics` / `smoke_subset`
- 生ログの場所（ローカルのみ）: `diag_logs_recheck/20261004-193503_7db82f62/rank0`

[設定](metadata.json) / [集計](metrics.json) / [生ファイルの識別情報](sources.json) / [diagnostic_summary.md](diagnostic_summary.md) / [state.yaml](state.yaml)

## 主要な結果

- スコア収集: 16バッチ / 64画像
- 候補評価: 10,026行、非有限Fitness 0行
- 枝刈り直後: Top-1 0.0000% / Top-5 0.0000% / loss 7.4803
- 削除数: `{'dim': 94, 'hidden': 0, 'head': 9}`

## 解釈上の注意

- 設定・集計のみを保存。重み・tensor・巨大な生ログは含めていません。
- 不完全な記録の終了理由は、ログだけから断定していません。
- ゼロ分母処理などの修正版。NaNデータを単に除外した実行ではありません。乱数固定・候補順も変更されており、精度差をバグ修正単独の効果とは断定できません。
- 少数画像の動作確認です。精度を研究上の性能値として比較しません。

## 再現性

実行時commit: `dc2ae2a563ebbcd51c8463f2c6fc03b041d0c951`、dirty: `True`。dirty=Trueならcommitだけでは当時の全コードを復元できません。ソースhashがある場合も、ソース本体はローカルarchiveにのみあります。

実行引数はmetadata.jsonのargvを参照。元ファイルとの対応はsources.jsonのpath・SHA-256で追跡できます。
