# 実行 20261001-122142_1ed34567

保存時点: 2026-10-11（Asia/Tokyo）

- 処理: `fintune.py`
- 状態: `finetuning_completed`
- 条件: `finetuning` / `full_dataset_requested`
- 生ログの場所（ローカルのみ）: `diag_logs/20261001-122142_1ed34567/rank0`

[設定](metadata.json) / [集計](metrics.json) / [生ファイルの識別情報](sources.json) / [epoch_metrics.csv](epoch_metrics.csv) / [config.yaml](config.yaml)

## 主要な結果

- 記録: 20/20 epoch、最後はepoch 19
- 最後の検証: Top-1 54.8501% / Top-5 79.5058%
- 最良Top-1: 54.8501%（epoch 19）
- 元の枝刈り実行: [20261001-114610_c49f9c88](../20261001-114610_c49f9c88/README.md)

## 解釈上の注意

- 設定・集計のみを保存。重み・tensor・巨大な生ログは含めていません。
- 不完全な記録の終了理由は、ログだけから断定していません。

## 再現性

実行時commit: `dc2ae2a563ebbcd51c8463f2c6fc03b041d0c951`、dirty: `True`。dirty=Trueならcommitだけでは当時の全コードを復元できません。ソースhashがある場合も、ソース本体はローカルarchiveにのみあります。

実行引数はmetadata.jsonのargvを参照。元ファイルとの対応はsources.jsonのpath・SHA-256で追跡できます。
