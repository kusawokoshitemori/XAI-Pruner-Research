# 実行 20261001-114610_c49f9c88

保存時点: 2026-10-11（Asia/Tokyo）

- 処理: `prune-ViT.py`
- 状態: `pruning_evaluation_completed`
- 条件: `original_behavior_with_diagnostics` / `full_dataset_requested`
- 生ログの場所（ローカルのみ）: `diag_logs/20261001-114610_c49f9c88/rank0`

[設定](metadata.json) / [集計](metrics.json) / [生ファイルの識別情報](sources.json) / [diagnostic_summary.md](diagnostic_summary.md) / [state.yaml](state.yaml)

## 主要な結果

- スコア収集: 1,137バッチ / 36,381画像
- 候補評価: 10,019行、非有限Fitness 10,019行
- 枝刈り直後: Top-1 0.1704% / Top-5 0.8026% / loss 7.1498
- 削除数: `{'dim': 78, 'hidden': 1426, 'head': 11}`

## 解釈上の注意

- 設定・集計のみを保存。重み・tensor・巨大な生ログは含めていません。
- 不完全な記録の終了理由は、ログだけから断定していません。
- Fitnessが非有限のため、候補順位・生存率から候補品質を評価できません。

## 再現性

実行時commit: `9caf6f865d5a41cf89fe684d41aa5c3417ddfb1a`、dirty: `True`。dirty=Trueならcommitだけでは当時の全コードを復元できません。ソースhashがある場合も、ソース本体はローカルarchiveにのみあります。

実行引数はmetadata.jsonのargvを参照。元ファイルとの対応はsources.jsonのpath・SHA-256で追跡できます。
