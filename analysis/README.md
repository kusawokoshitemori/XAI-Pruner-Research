# ログ・実験資料の案内

2026-10-19整理。今回のコミットは文書と編集履歴の参照資料のみ。既存の未コミット計算コード・生ログ・重みは含めない。

| 知りたいこと | 読む資料 |
|---|---|
| 何を疑って、何を記録したか | [目的別ログ一覧](logging_purpose_guide.md) |
| 引数・出力ファイル・現在のイベント仕様 | [diag/README](../diag/README.md) |
| 全ファイル・CSV列・イベント項目と#003〜#017の対応 | [詳細目録](diagnostic_log_inventory.md)（主対象は10/1。件数は当時の固定値） |
| 元のNaNを含む実行の分析 | [10/1実行の分析](diagnostic_log_analysis_20261001.md) |
| 数値処理を変更した別実験 | [10/4修正版の結果](rerun_results_20261004.md) |
| 何を変更し、どう戻したか | [復元記録](restoration_20261004/findings_and_restoration.md)、[操作履歴](restoration_20261004/reversal_journal.json) |
| 最新の公開済み固定中間結果・環境・学習曲線 | [10/18中間報告](../results/reports/20261018_tiny_interim/README.md)（epoch60時点） |
| 実行ごとの小さな保存結果 | [results一覧](../results/README.md) |
| val50,000枚への振り分け修復 | [修復記録](imagenet_val_repair_20261013.md) |

生ログの場所は`diag_logs/`、`diag_logs_recheck/`、`diag_logs_reproduction/`、重みと訓練ログは`output/`。これらはローカルのみで、本文中の絶対パスもローカル参照。GitHubで閲覧する場合はresultsの固定集計・設定・出典hashを使う。

詳細目録は10/4時点の調査であり、後から追加されたLayerNorm監視等はdiag/READMEと目的別一覧を併読する。資料にある仮説や追加実験案は、実施済み・実測結果を意味しない。
