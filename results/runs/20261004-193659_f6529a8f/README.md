# 実行 20261004-193659_f6529a8f

保存時点: 2026-10-11（Asia/Tokyo）

- 処理: `prune-ViT.py`
- 状態: `recorded_workflow_incomplete`
- 条件: `numerical_fixes_and_diagnostics` / `full_dataset_requested`
- 生ログの場所（ローカルのみ）: `diag_logs_recheck/20261004-193659_f6529a8f/rank0`

[設定](metadata.json) / [集計](metrics.json) / [生ファイルの識別情報](sources.json) / [source_code_hashes.json](source_code_hashes.json)

## 主要な結果

- スコア収集: 0バッチ / 0画像

## 解釈上の注意

- 設定・集計のみを保存。重み・tensor・巨大な生ログは含めていません。
- 不完全な記録の終了理由は、ログだけから断定していません。
- ゼロ分母処理などの修正版。NaNデータを単に除外した実行ではありません。乱数固定・候補順も変更されており、精度差をバグ修正単独の効果とは断定できません。

## 再現性

実行時commit: `dc2ae2a563ebbcd51c8463f2c6fc03b041d0c951`、dirty: `True`。dirty=Trueならcommitだけでは当時の全コードを復元できません。ソースhashがある場合も、ソース本体はローカルarchiveにのみあります。

実行引数はmetadata.jsonのargvを参照。元ファイルとの対応はsources.jsonのpath・SHA-256で追跡できます。
