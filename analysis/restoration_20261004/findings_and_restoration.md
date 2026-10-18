> 生ログ・重み・ソースZIPへの参照はローカルのみの生ログ資料です。GitHubには含めません。文書の確認日と実行条件を併読してください。

# 今回分かったことと、観測専用への復元

## 復元の根拠・範囲

復元対象はmainではなく、この会話でAIが数値・探索処理を変更する直前の作業コピーの挙動。会話の実行ツールには個別の置換コードと変更前の読取結果があり、そこからAIが追加した変更を逆適用した。mainとの差分全体の取消・git reset・git checkoutは行っていない。

変更前作業コピーの全ファイルを一括保存したsnapshotはない。このため、当時の全ファイルのbyte単位一致を一括証明することはできない。ただし今回AIが変更した計算処理・乱数処理・停止条件の変更箇所と元の式・呼出しは編集記録で特定でき、個別復元した。見知らぬ差分をmainへ戻す操作はしていない。

- [復元前の修正版コードarchive](modified_code_before_restore.zip)
- [復元前コードのhash](modified_code_hashes.json)
- [個別復元操作（before/after全文）](reversal_journal.json)
- [編集記録から再構成した変更前LRPルールの検証参照](reference_rules_before_intervention.py.txt)
- [修正版GPU実験の実行時ソースarchive](../../diag_logs_recheck/20261004-194102_3030b41f/rank0/source_code.zip)

## 戻した動作

| AIが加えた変更 | 復元後 |
|---|---|
| sign(0)にもepsilonを加える | 元の `epsilon * torch.sign(input)`。ゼロ分母はゼロのまま |
| 全ゼロ/相殺を特別処理するTransformer Filter | 元の `R * (original_sum / selected_sum)`。NaNもそのまま発生し得る |
| 非有限スコアで停止 | 記録のみ。スコア更新を元どおり続行 |
| 非有限Fitnessで停止 | 記録のみ。元のargsortで選択を続行 |
| Python randomをseed固定 | 固定しない。以前の乱数消費を維持 |
| 親プールをsorted tupleへ変換 | 元の `random.sample(self.candidates, ...)` |
| 候補をsortedで列挙 | 元の `list(self.candidates)` |
| 動作確認用データ切詰め引数 | 引数・切詰め処理を削除 |

LayerNormの追加tag・監視呼出しは観測用として残す。epsilonの値、LayerNormのforward/backwardの数式、返却関連度は元の処理のまま。計算対象をコピー/要約してログに記録する。新規/累積score監視、異常batch snapshot、失敗の記録、集計の警告も残す。異常を検出してもスコアの修復や候補の並べ替え・除外・停止はしない。

**元のバグを含む挙動に戻すため、NaNが再び発生するのは想定どおり。目的は元手法の観測であり、修正による性能改善ではない。** Python 3.10のXAIPruner環境を使う。元と同じrandom.sample(set)に戻したため、Python 3.11以降への互換修正も撤回した。

## 分かったこと（実測と解釈を分離）

### 1. 元実行の評価値が成立していなかった

元実行では272/624/766/771/934番バッチでdimensionスコア192個がNaNになった。累積は272番から回復せず、最終192個も全NaN。候補評価10,019行すべてのFitnessがNaN。

根拠: [最初のNaNスコア行](../../diag_logs/20261001-114610_c49f9c88/rank0/score_accumulation.csv:6826)、[元の候補評価](../../diag_logs/20261001-114610_c49f9c88/rank0/candidate_evaluations.csv)、[元の保存スコア](../../diag_logs/20261001-114610_c49f9c88/rank0/scores_momentum.npz)。

これはFitnessの代理指標としての良さを検証する前の実装上の問題。NaNの候補順位を重要度による良し悪しとして扱えない。

### 2. ゼロ分母がNaNに関与する強い証拠

修正版GPU実行では、最終LayerNormのraw出力がゼロになるバッチが272/624/766/771/934で、元のNaNバッチと完全一致。修正版では同箇所の返却関連度と全スコア・Fitnessが有限になった。

根拠: [272番の最終LayerNorm監視](../../diag_logs_recheck/20261004-194102_3030b41f/rank0/events.jsonl:7718)、[修正版の全スコア監視](../../diag_logs_recheck/20261004-194102_3030b41f/rank0/events.jsonl)、[修正版の候補評価](../../diag_logs_recheck/20261004-194102_3030b41f/rank0/candidate_evaluations.csv)。修正版の式は [実行時ソースarchive](../../diag_logs_recheck/20261004-194102_3030b41f/rank0/source_code.zip) に保存。

解釈: 元のstabilizeはsign(0)=0なのでゼロを除去しない。ゼロ関連度/ゼロ分母でも0/0はNaNになる。この機序は上記一致と局所テストが強く支持する。ただし当該5バッチを旧式と修正式で個別再実行する対照実験は未実施。

### 3. patch Filterが再スケーリング自身でNaNを作っていた

元の先頭2バッチでは、受信関連度が全ゼロにもかかわらず0/0の再スケーリングが行われ、一時NaNを作ってからnan_to_numでゼロへ修復していた。

根拠: [元のFilterイベント](../../diag_logs/20261001-114610_c49f9c88/rank0/events.jsonl:125)、[元の異常snapshot](../../diag_logs/20261001-114610_c49f9c88/rank0/snapshots/0124_filter_cnn_nonfinite_patch_embed.proj.filter.pt)。

これは#015・#012の材料。dimension NaNと同じ発生場所ではないので、二つを同じ原因として扱わない。

### 4. 数値修正は精度改善を証明しなかった

修正版の全件GPU実験は、画像hash36,381個すべてが元実行と同じ順番で一致。最終スコアと候補Fitnessは有限でマスク整合性も成立したが、fine-tuning前Top-1は0.0907%、Top-5は0.6294%と低かった。元のTop-1は0.1704%。

根拠: [修正版の最終評価](../../diag_logs_recheck/20261004-194102_3030b41f/rank0/events.jsonl:31406)、[元の最終評価](../../diag_logs/20261001-114610_c49f9c88/rank0/events.jsonl:359)、[元画像hash](../../diag_logs/20261001-114610_c49f9c88/rank0/initial_relevance.csv)、[修正版画像hash](../../diag_logs_recheck/20261004-194102_3030b41f/rank0/initial_relevance.csv)。

修正版は数式・Python乱数・候補順を複数変更した別条件の実験なので、差を単一修正の効果として解釈しない。保存モデルはそのまま残しているが、元手法のモデルとして混ぜない。

### 5. その他の研究上の観測

交叉・変異の新規性、残差相殺、Filterの除去量、biasの符号反転などは元実行のログで確認済み。疑問ごとの根拠・未検証部分は以下を参照。

- [元実行の分析](../diagnostic_log_analysis_20261001.md)
- [ファイル・イベント・全列と疑問番号の目録](../diagnostic_log_inventory.md)
- [修正版実験の記録](../rerun_results_20261004.md)

## 検証と限界

[観測専用の検証テスト](../../tests/test_observation_only.py) 3件が通過。変更前ルールの参照と、監視有効状態のLayerNorm/Filterのforward/backwardを、通常値・ゼロ・相殺入力で比較し、NaN/Infを含め完全一致を確認。Python/NumPy/PyTorch CPU乱数状態も一致。非有限scoreを変更しないこと、NaN Fitnessでも元の選択を続けることを確認。

これは局所処理と選択の検証であり、GPU全モデル・全datasetのログあり/なしの精度一致実験はまだしていない。ログには時間・メモリ・同期負荷がある。Python乱数は元から未固定なので、独立した実行同士で候補や精度の一致は保証しない。

観測専用の全件GPU再実行は、この復元作業では開始していない。`tools/rerun_diag_vit.sh` の出力先は新規 `output/diag_observation_<timestamp>`、診断は `diag_logs_observation/` へ変更した。以前の修正版実験の結果を上書きしない。
