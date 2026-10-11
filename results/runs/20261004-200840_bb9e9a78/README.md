# 実行 20261004-200840_bb9e9a78

保存時点: 2026-10-11（Asia/Tokyo）

- 処理: `fintune.py`
- 状態: `finetuning_terminated_sigkill`
- 条件: `finetuning` / `full_dataset_requested`
- 生ログの場所（ローカルのみ）: `diag_logs_recheck/20261004-200840_bb9e9a78/rank0`

[設定](metadata.json) / [集計](metrics.json) / [生ファイルの識別情報](sources.json) / [epoch_metrics.csv](epoch_metrics.csv) / [config.yaml](config.yaml)

## 主要な結果

- 記録: 14/20 epoch、最後はepoch 13
- 最後の検証: Top-1 47.9866% / Top-5 74.1321%
- 最良Top-1: 47.9866%（epoch 13）
- 元の枝刈り実行: [20261004-194102_3030b41f](../20261004-194102_3030b41f/README.md)

## 解釈上の注意

- 設定・集計のみを保存。重み・tensor・巨大な生ログは含めていません。
- 不完全な記録の終了理由は、ログだけから断定していません。
- ユーザー提供の端末ログで、2026-10-08 19:51:47にepoch 14の途中（最後の表示4330/5004）でSIGKILL / exitcode -9を確認。原因は未確定。詳細は[終了記録](termination_report.json)。

## 再現性

実行時commit: `dc2ae2a563ebbcd51c8463f2c6fc03b041d0c951`、dirty: `True`。dirty=Trueならcommitだけでは当時の全コードを復元できません。ソースhashがある場合も、ソース本体はローカルarchiveにのみあります。

実行引数はmetadata.jsonのargvを参照。元ファイルとの対応はsources.jsonのpath・SHA-256で追跡できます。
