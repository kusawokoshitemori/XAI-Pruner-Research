# 実験条件・変更点・ログの所在一覧

確認日：2026-10-19。対象はResearch作業コピーの診断ディレクトリに残る18件のmetadata。別ディレクトリ・別PC・ログ無効の実行まで網羅した一覧ではない。

**Research側の18件は全てDeiT-Tiny。ただし別の元コードcheckoutにBaseの実行結果を発見した。** 下記B01とチャット証拠の補足も参照。Research側だけの検索でBase未実行と結論しない。 コマンド例と実行実績を区別する。epochは指定値であり完了数とは限らない。batchは1プロセス当たり。

## 条件の変更の流れ

| 日付・系列 | 実行した条件・変更 | 元ログの場所 |
|---|---|---|
| 10/1 元挙動 | 枝刈りbatch32、削減目標50%。fine-tuning batch256、300/0/10/20 epochの開始記録 | diag_logs/ |
| 10/4 CPU動作確認 | 数値修正版。batch4、8画像／64画像に限定 | diag_logs_recheck/20261004-193344_*、193503_* |
| 10/4 全件試行 | batch32、削減目標50%、CPU途中試行→GPU全件。LRP分母・Filter・Python乱数・候補順などを変更 | diag_logs_recheck/20261004-193659_*、194102_* |
| 10/4 修正版モデル訓練 | batch256、20 epoch。epoch14途中でOOM終了 | diag_logs_recheck/20261004-200840_*、output/.../finetune_v1/ |
| 10/11 再開試行 | batch256、通算20 epoch指定。複数の開始metadata。同じ再開先を使用 | diag_logs_recheck/20261011-*、output/.../finetune_resume_v2/ |
| 10/13 元LRP挙動の観測 | 関連度batch128、削減目標24.4%、EVO_IMNETでsubImageNet10万枚。valは修復後5万枚 | diag_logs_reproduction/20261013-133112_8236a79c/ |
| 10/13 現在の訓練 | batch128、300 epoch、lr1e-3→1e-5、warmup/cooldown0 | diag_logs_reproduction/20261013-140236_906ce367/、output/tiny_reproduction_20261013-133056/finetune/ |

数値修正版は単なるログ追加の実験とは別。10/13は元LRPの数式でも、EVO_IMNETのデータ選択が元公開コードの既定動作と異なる。比較条件の詳細は[中間報告](../results/reports/20261018_tiny_interim/README.md)。

## 全開始記録

| ID | run ID | 処理 | batch | 指定epoch | device | 削減率目標 | 確認できる状態 |
|---|---|---|---:|---:|---|---:|---|
| [E01](#e01) | 20261001-113843_c49f9c88 | prune-ViT.py | 32 | — | cuda | 0.5 | recorded_workflow_incomplete |
| [E02](#e02) | 20261001-114610_c49f9c88 | prune-ViT.py | 32 | — | cuda | 0.5 | 枝刈り・最終評価まで記録あり |
| [E03](#e03) | 20261001-115750_3170f88a | fintune.py | 256 | 300 | cuda | — | metadata_only |
| [E04](#e04) | 20261001-120010_6c3f73ad | fintune.py | 256 | 0 | cuda | — | metadata_only |
| [E05](#e05) | 20261001-120033_cbb12660 | fintune.py | 256 | 300 | cuda | — | diagnostics_only |
| [E06](#e06) | 20261001-120342_ddc1852e | fintune.py | 256 | 300 | cuda | — | metadata_only |
| [E07](#e07) | 20261001-121720_6c82be4b | fintune.py | 256 | 10 | cuda | — | metadata_only |
| [E08](#e08) | 20261001-122142_1ed34567 | fintune.py | 256 | 20 | cuda | — | 記録上は学習完了 |
| [E09](#e09) | 20261004-193344_258f7af3 | prune-ViT.py | 4 | — | cpu | 0.5 | 枝刈り・最終評価まで記録あり |
| [E10](#e10) | 20261004-193503_7db82f62 | prune-ViT.py | 4 | — | cpu | 0.5 | 枝刈り・最終評価まで記録あり |
| [E11](#e11) | 20261004-193659_f6529a8f | prune-ViT.py | 32 | — | cpu | 0.5 | recorded_workflow_incomplete |
| [E12](#e12) | 20261004-194102_3030b41f | prune-ViT.py | 32 | — | cuda | 0.5 | 枝刈り・最終評価まで記録あり |
| [E13](#e13) | 20261004-200840_bb9e9a78 | fintune.py | 256 | 20 | cuda | — | epoch14途中でOOM終了 |
| [E14](#e14) | 20261011-125131_b72be03f | fintune.py | 256 | 20 | cuda | — | metadata_recorded_only |
| [E15](#e15) | 20261011-125319_b72be03f | fintune.py | 256 | 20 | cuda | — | metadata_recorded_only |
| [E16](#e16) | 20261011-125333_b72be03f | fintune.py | 256 | 20 | cuda | — | 検証記録あり・終了未確認 |
| [E17](#e17) | 20261013-133112_8236a79c | prune-ViT.py | 128 | — | cuda | 0.244 | 枝刈り・最終評価まで記録あり |
| [E18](#e18) | 20261013-140236_906ce367 | fintune.py | 128 | 300 | cuda | — | 検証記録あり・終了未確認 |

## 実行ごとの正確な場所

以下のコード表記パスはリポジトリルートからの相対パス。生ログ・重みはローカルにのみ存在し、GitHubへ全量を入れていない。JSON版には全実行引数・argv・最終イベントも収録： [experiment_ledger.json](experiment_ledger.json)。

### E01

- run ID：`20261001-113843_c49f9c88`
- モデル・区分：`deit_tiny_patch16_224` / 元挙動の観測
- 実行条件の一次情報：`diag_logs/20261001-113843_c49f9c88/rank0/metadata.json`
- 内部イベント：`diag_logs/20261001-113843_c49f9c88/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny`
- 読み込んだ重み：`./checkpoint/deit_tiny_patch16_224.pth`
- データ：`/home/nakata/data_dl/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- スコアとNaN：`diag_logs/20261001-113843_c49f9c88/rank0/score_accumulation.csv`、`scores_momentum.npz`（未到達なら存在しない）
- EAの試行・候補：同ディレクトリの`ea_attempts.csv`・`candidate_evaluations.csv`（未到達なら存在しない）
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261001-113843_c49f9c88/README.md)（以前の保存時点）。

### E02

- run ID：`20261001-114610_c49f9c88`
- モデル・区分：`deit_tiny_patch16_224` / 元挙動の観測
- 実行条件の一次情報：`diag_logs/20261001-114610_c49f9c88/rank0/metadata.json`
- 内部イベント：`diag_logs/20261001-114610_c49f9c88/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny`
- 読み込んだ重み：`./checkpoint/deit_tiny_patch16_224.pth`
- データ：`/home/nakata/data_dl/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- スコアとNaN：`diag_logs/20261001-114610_c49f9c88/rank0/score_accumulation.csv`、`scores_momentum.npz`（未到達なら存在しない）
- EAの試行・候補：同ディレクトリの`ea_attempts.csv`・`candidate_evaluations.csv`（未到達なら存在しない）
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261001-114610_c49f9c88/README.md)（以前の保存時点）。

### E03

- run ID：`20261001-115750_3170f88a`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs/20261001-115750_3170f88a/rank0/metadata.json`
- 内部イベント：`diag_logs/20261001-115750_3170f88a/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny/finetune`
- 読み込んだ重み：`./output/tiny/checkpoint_pruned.pth`
- データ：`./data/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261001-115750_3170f88a/README.md)（以前の保存時点）。
- 注意：output_dirを後続実行と共有。現在のlog.txtをこの試行の結果とは扱わない。

### E04

- run ID：`20261001-120010_6c3f73ad`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs/20261001-120010_6c3f73ad/rank0/metadata.json`
- 内部イベント：`diag_logs/20261001-120010_6c3f73ad/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny/finetune_smoke`
- 読み込んだ重み：`./output/tiny/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261001-120010_6c3f73ad/README.md)（以前の保存時点）。

### E05

- run ID：`20261001-120033_cbb12660`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs/20261001-120033_cbb12660/rank0/metadata.json`
- 内部イベント：`diag_logs/20261001-120033_cbb12660/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny/finetune_check`
- 読み込んだ重み：`./output/tiny/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261001-120033_cbb12660/README.md)（以前の保存時点）。

### E06

- run ID：`20261001-120342_ddc1852e`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs/20261001-120342_ddc1852e/rank0/metadata.json`
- 内部イベント：`diag_logs/20261001-120342_ddc1852e/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny/finetune`
- 読み込んだ重み：`./output/tiny/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261001-120342_ddc1852e/README.md)（以前の保存時点）。
- 注意：output_dirを後続実行と共有。現在のlog.txtをこの試行の結果とは扱わない。

### E07

- run ID：`20261001-121720_6c82be4b`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs/20261001-121720_6c82be4b/rank0/metadata.json`
- 内部イベント：`diag_logs/20261001-121720_6c82be4b/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny/finetune`
- 読み込んだ重み：`./output/tiny/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261001-121720_6c82be4b/README.md)（以前の保存時点）。
- 注意：output_dirを後続実行と共有。現在のlog.txtをこの試行の結果とは扱わない。

### E08

- run ID：`20261001-122142_1ed34567`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs/20261001-122142_1ed34567/rank0/metadata.json`
- 内部イベント：`diag_logs/20261001-122142_1ed34567/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny/finetune`
- 読み込んだ重み：`./output/tiny/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- この保存先の現在の記録：20検証、最後epoch 19 / Top-1 54.8501%、最高 54.8501%（epoch 19）。記録取得日10/19の値で、以前の固定レポートを更新するものではない。
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261001-122142_1ed34567/README.md)（以前の保存時点）。

### E09

- run ID：`20261004-193344_258f7af3`
- モデル・区分：`deit_tiny_patch16_224` / 数値修正版
- 実行条件の一次情報：`diag_logs_recheck/20261004-193344_258f7af3/rank0/metadata.json`
- 内部イベント：`diag_logs_recheck/20261004-193344_258f7af3/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/diag_recheck_cpu`
- 読み込んだ重み：`./checkpoint/deit_tiny_patch16_224.pth`
- データ：`/home/nakata/data_dl/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cpu`
- 動作確認の限定画像数：8。性能値の比較には使わない。
- スコアとNaN：`diag_logs_recheck/20261004-193344_258f7af3/rank0/score_accumulation.csv`、`scores_momentum.npz`（未到達なら存在しない）
- EAの試行・候補：同ディレクトリの`ea_attempts.csv`・`candidate_evaluations.csv`（未到達なら存在しない）
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261004-193344_258f7af3/README.md)（以前の保存時点）。

### E10

- run ID：`20261004-193503_7db82f62`
- モデル・区分：`deit_tiny_patch16_224` / 数値修正版
- 実行条件の一次情報：`diag_logs_recheck/20261004-193503_7db82f62/rank0/metadata.json`
- 内部イベント：`diag_logs_recheck/20261004-193503_7db82f62/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/diag_recheck_cpu64`
- 読み込んだ重み：`./checkpoint/deit_tiny_patch16_224.pth`
- データ：`/home/nakata/data_dl/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cpu`
- 動作確認の限定画像数：64。性能値の比較には使わない。
- スコアとNaN：`diag_logs_recheck/20261004-193503_7db82f62/rank0/score_accumulation.csv`、`scores_momentum.npz`（未到達なら存在しない）
- EAの試行・候補：同ディレクトリの`ea_attempts.csv`・`candidate_evaluations.csv`（未到達なら存在しない）
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261004-193503_7db82f62/README.md)（以前の保存時点）。

### E11

- run ID：`20261004-193659_f6529a8f`
- モデル・区分：`deit_tiny_patch16_224` / 数値修正版
- 実行条件の一次情報：`diag_logs_recheck/20261004-193659_f6529a8f/rank0/metadata.json`
- 内部イベント：`diag_logs_recheck/20261004-193659_f6529a8f/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/diag_recheck_full_cpu`
- 読み込んだ重み：`./checkpoint/deit_tiny_patch16_224.pth`
- データ：`/home/nakata/data_dl/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cpu`
- スコアとNaN：`diag_logs_recheck/20261004-193659_f6529a8f/rank0/score_accumulation.csv`、`scores_momentum.npz`（未到達なら存在しない）
- EAの試行・候補：同ディレクトリの`ea_attempts.csv`・`candidate_evaluations.csv`（未到達なら存在しない）
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261004-193659_f6529a8f/README.md)（以前の保存時点）。

### E12

- run ID：`20261004-194102_3030b41f`
- モデル・区分：`deit_tiny_patch16_224` / 数値修正版
- 実行条件の一次情報：`diag_logs_recheck/20261004-194102_3030b41f/rank0/metadata.json`
- 内部イベント：`diag_logs_recheck/20261004-194102_3030b41f/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/diag_recheck_full_20261004-194058`
- 読み込んだ重み：`./checkpoint/deit_tiny_patch16_224.pth`
- データ：`/home/nakata/data_dl/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- スコアとNaN：`diag_logs_recheck/20261004-194102_3030b41f/rank0/score_accumulation.csv`、`scores_momentum.npz`（未到達なら存在しない）
- EAの試行・候補：同ディレクトリの`ea_attempts.csv`・`candidate_evaluations.csv`（未到達なら存在しない）
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261004-194102_3030b41f/README.md)（以前の保存時点）。

### E13

- run ID：`20261004-200840_bb9e9a78`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs_recheck/20261004-200840_bb9e9a78/rank0/metadata.json`
- 内部イベント：`diag_logs_recheck/20261004-200840_bb9e9a78/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/diag_recheck_full_20261004-194058/finetune_v1`
- 読み込んだ重み：`./output/diag_recheck_full_20261004-194058/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- この保存先の現在の記録：14検証、最後epoch 13 / Top-1 47.9866%、最高 47.9866%（epoch 13）。記録取得日10/19の値で、以前の固定レポートを更新するものではない。
- コミット済みの小さな保存結果：[このrunのREADME](../results/runs/20261004-200840_bb9e9a78/README.md)（以前の保存時点）。

### E14

- run ID：`20261011-125131_b72be03f`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs_recheck/20261011-125131_b72be03f/rank0/metadata.json`
- 内部イベント：`diag_logs_recheck/20261011-125131_b72be03f/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/diag_recheck_full_20261004-194058/finetune_resume_v2`
- 読み込んだ重み：`./output/diag_recheck_full_20261004-194058/finetune_v1/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- 注意：output_dirを後続実行と共有。現在のlog.txtをこの試行の結果とは扱わない。

### E15

- run ID：`20261011-125319_b72be03f`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs_recheck/20261011-125319_b72be03f/rank0/metadata.json`
- 内部イベント：`diag_logs_recheck/20261011-125319_b72be03f/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/diag_recheck_full_20261004-194058/finetune_resume_v2`
- 読み込んだ重み：`./output/diag_recheck_full_20261004-194058/finetune_v1/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- 注意：output_dirを後続実行と共有。現在のlog.txtをこの試行の結果とは扱わない。

### E16

- run ID：`20261011-125333_b72be03f`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs_recheck/20261011-125333_b72be03f/rank0/metadata.json`
- 内部イベント：`diag_logs_recheck/20261011-125333_b72be03f/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/diag_recheck_full_20261004-194058/finetune_resume_v2`
- 読み込んだ重み：`./output/diag_recheck_full_20261004-194058/finetune_v1/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- この保存先の現在の記録：6検証、最後epoch 19 / Top-1 51.0761%、最高 51.0761%（epoch 19）。記録取得日10/19の値で、以前の固定レポートを更新するものではない。

### E17

- run ID：`20261013-133112_8236a79c`
- モデル・区分：`deit_tiny_patch16_224` / 元LRP挙動の観測（EVO_IMNET選択あり）
- 実行条件の一次情報：`diag_logs_reproduction/20261013-133112_8236a79c/rank0/metadata.json`
- 内部イベント：`diag_logs_reproduction/20261013-133112_8236a79c/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny_reproduction_20261013-133056`
- 読み込んだ重み：`./checkpoint/deit_tiny_patch16_224.pth`
- データ：`/mnt/data1/imagenet` / `data_set=EVO_IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- スコアとNaN：`diag_logs_reproduction/20261013-133112_8236a79c/rank0/score_accumulation.csv`、`scores_momentum.npz`（未到達なら存在しない）
- EAの試行・候補：同ディレクトリの`ea_attempts.csv`・`candidate_evaluations.csv`（未到達なら存在しない）

### E18

- run ID：`20261013-140236_906ce367`
- モデル・区分：`deit_tiny_patch16_224` / fine-tuning（元モデルはresume参照）
- 実行条件の一次情報：`diag_logs_reproduction/20261013-140236_906ce367/rank0/metadata.json`
- 内部イベント：`diag_logs_reproduction/20261013-140236_906ce367/rank0/events.jsonl`（空なら内部イベントなし）
- 出力先：`./output/tiny_reproduction_20261013-133056/finetune`
- 読み込んだ重み：`./output/tiny_reproduction_20261013-133056/checkpoint_pruned.pth`
- データ：`/mnt/data1/imagenet` / `data_set=IMNET`
- GPU/プロセス数設定：`world_size=1`、requested device=`cuda`
- 学習曲線・設定：出力先の`log.txt`・`config.yaml`。共有先では後続実行の内容に注意。
- この保存先の現在の記録：73検証、最後epoch 72 / Top-1 39.7300%、最高 39.7300%（epoch 72）。記録取得日10/19の値で、以前の固定レポートを更新するものではない。

## 対応する過去資料

- [ログ追加の目的](logging_purpose_guide.md)：なぜ記録したか。実験条件の台帳とは役割が違う。
- [10/1元挙動の分析](diagnostic_log_analysis_20261001.md)、[10/4修正版の分析](rerun_results_20261004.md)。
- [実行単位の保存済み結果](../results/README.md)、[epoch60時点の固定中間報告](../results/reports/20261018_tiny_interim/README.md)。
- 各実行の開始metadataは残っていても、consoleやソースsnapshotが必ずあるわけではない。dirty=Trueのcommitと引数だけで実行時コード全体を復元できるとは限らない。

## B01：元コードcheckoutに残っていたBase実験

- 場所：`/home/nakata/code/XAI-Pruner/output/finetune/`（Researchとは別）。
- config.yaml：DeiT-Base、batch32、world_size1、10 epoch、lr1e-3、warmup5、cooldown10。
- 実測：epoch0〜9の10検証と終了時間18:45:23を記録。最終Top-1 53.98697%、最高67.34284%（epoch1）。42,601,421 parameters。
- 設定作成とログ更新のmtimeは9/27〜28。正確な開始/終了時刻の代わりには使わない。
- cfg=`./output/state.yaml`、resume=`./output/checkpoint_pruned.pth`、data_path=`./data/imagenet`。これらの相対パスは元コードcheckout基準。
- [固定した設定](base_previous_experiment/config.yaml)、[学習CSV](base_previous_experiment/epoch_metrics.csv)、[構造](base_previous_experiment/state.yaml)、[根拠・hash](base_previous_experiment/facts.json)。重みはローカルのみ。
- 枝刈り時のコマンド・batch・目標削減率・NaN有無はこの学習設定からは確定できない。実行当時のval枚数も未記録で、現在のTinyの5万枚評価と直接比較しない。

## チャット内の提案と実行報告

[チャットを調べた補足](chat_experiment_notes.md)を参照。30 epoch等の提案を、完了した実験として数えない。
