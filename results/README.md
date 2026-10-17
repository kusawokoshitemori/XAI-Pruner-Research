# 実験結果の保存記録

保存日: 2026-10-11。各実行は `runs/<run_id>/` に分離し、実行単位でコミットしています。

重み・生ログ・テンソル・ソースZIPは保存せず、設定、集計、元ファイルのサイズとSHA-256を保存します。元ファイルはローカルに残しています。集計値に現れる `NaN` は文字列として明示し、正常な数値へ置き換えていません。

## 実行一覧

| 実行 | 処理 | 条件 | 保存時点の状態 |
|---|---|---|---|
| [20261001-113843_c49f9c88](runs/20261001-113843_c49f9c88/README.md) | prune-ViT.py | original_behavior_with_diagnostics | recorded_workflow_incomplete |
| [20261001-114610_c49f9c88](runs/20261001-114610_c49f9c88/README.md) | prune-ViT.py | original_behavior_with_diagnostics | pruning_evaluation_completed |
| [20261001-115750_3170f88a](runs/20261001-115750_3170f88a/README.md) | fintune.py | finetuning | metadata_only |
| [20261001-120010_6c3f73ad](runs/20261001-120010_6c3f73ad/README.md) | fintune.py | finetuning | metadata_only |
| [20261001-120033_cbb12660](runs/20261001-120033_cbb12660/README.md) | fintune.py | finetuning | diagnostics_only |
| [20261001-120342_ddc1852e](runs/20261001-120342_ddc1852e/README.md) | fintune.py | finetuning | metadata_only |
| [20261001-121720_6c82be4b](runs/20261001-121720_6c82be4b/README.md) | fintune.py | finetuning | metadata_only |
| [20261001-122142_1ed34567](runs/20261001-122142_1ed34567/README.md) | fintune.py | finetuning | finetuning_completed |
| [20261004-193344_258f7af3](runs/20261004-193344_258f7af3/README.md) | prune-ViT.py | numerical_fixes_and_diagnostics | pruning_evaluation_completed |
| [20261004-193503_7db82f62](runs/20261004-193503_7db82f62/README.md) | prune-ViT.py | numerical_fixes_and_diagnostics | pruning_evaluation_completed |
| [20261004-193659_f6529a8f](runs/20261004-193659_f6529a8f/README.md) | prune-ViT.py | numerical_fixes_and_diagnostics | recorded_workflow_incomplete |
| [20261004-194102_3030b41f](runs/20261004-194102_3030b41f/README.md) | prune-ViT.py | numerical_fixes_and_diagnostics | pruning_evaluation_completed |
| [20261004-200840_bb9e9a78](runs/20261004-200840_bb9e9a78/README.md) | fintune.py | finetuning | finetuning_terminated_sigkill |

## 比較の注意

- 元の枝刈り実行は全候補のFitnessがNaN。候補順位の品質を評価できません。
- 数値修正版はゼロ分母処理等を変えた条件です。NaN画像を除外した実行とは区別します。乱数固定や候補順も違うため、性能差を数値修正だけに帰属できません。
- 元のfinetuneは20エポック完了、最終Top-1 54.8501%。修正版に続くfinetuneは14/20エポックの検証記録があり、epoch 14の途中でSIGKILL終了。最後のTop-1 47.9866%。途中値と完了値は直接の最終性能比較に使いません。
- `metadata_only` は当該実行の精度が確定できない記録。共有output_dirに残る後続実行の精度を誤って割り当てていません。
- `smoke_subset` は少数画像の動作確認。`recorded_workflow_incomplete` は完了を確認できない記録で、停止原因は断定していません。
- 今回の結果コミットには既存の訓練・枝刈りコードの未コミット変更を含めていません。

## 集計の再生成

NumPyが利用できる環境で、リポジトリのルートから実行します。生ログが更新されていれば記録も更新されます。再生成後は変更があった実行のみ別コミットにしてください。

```bash
python tools/archive_experiment_results.py --as-of-date 2026-10-11
```

## 2026-10-18の中間報告

[DeiT-Tiny：epoch 60終了時点の中間報告](reports/20261018_tiny_interim/README.md)には、学習曲線、実行設定・環境、NaN/EAの観測、元コード・論文との条件差と未検証事項を収録しています。途中結果の固定記録であり、300 epochの最終結果ではありません。
