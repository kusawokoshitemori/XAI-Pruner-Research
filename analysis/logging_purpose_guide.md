# 何のためにどのログを追加したか

入口は[案内](README.md)。疑問への「対応」は一部を観察可能という意味。因果・精度改善を証明した意味ではない。

| 調べる目的 | 追加した記録 | 対応する疑問・限界 |
|---|---|---|
| EAが新しい候補を作っているか | ea_attempts.csv、ea_operator_calls.csv：試行、親、変異遺伝子数、合法性、親/子内重複、新規候補/削除数マスク | #003/#007。性能優位には同コスト対照が必要 |
| どの生成方式が選択へ残るか | candidate_evaluations.csv、ea_generation：生成源、選択、生存率、順位、制約量 | #003/#007。NaN Fitnessでは候補品質は判定できない |
| 種類間の尺度・要素数効果 | component_scores.csv：S、n、g、S/n、S/g、種類別Fitness内訳 | #004。単独削除影響の較正実験はない |
| 代理Fitnessと実モデルを追跡 | 候補CSV、final_consistency、model_build、枝刈り直後の最終評価 | #005/#007。候補ごとの実測性能・相互作用は未収録 |
| 固定順位・削除集合の整合性 | score_version、候補ID、削除数由来mask ID、最終mask hash、実体寸法 | #006と実装確認。A/B/AB介入はしていない |
| 残差累積と相殺 | residual_add：shortcut/functionalのL2、RMS、内積、cos、配分関連度、分母 | #008/#011。枝削除の因果的寄与とは別 |
| Gateの動作 | gate：受信と返却関連度 | #009。Gate単独の消融結果はない |
| BN規則の前提 | batchnorm_config：モードと統計設定 | #010。TinyにはBNがなくCNNの比較が必要 |
| Filteringの選別と再増幅 | filter_runtime_config、filter_rescale：保持率、マスク前/後/再スケール後、係数、分母、符号、非有限数 | #013〜#016。詳細バッチ限定、分母と後続演算の対応には制約あり |
| bias再分配と符号反転 | linear_lrp/conv2d_lrp：修復前後、データ項、bias項、相殺、符号反転、保存 | #013/#017。順位・精度への因果は未検証 |
| momentumとデータ順の影響 | initial_relevance.csv、score_batches.csv、score_accumulation.csv、各scores NPZ、score_aggregation_comparison | #004/#006と再現性。単純平均は診断専用で選択に使わない |
| 設定・実行の取り違え防止 | metadata.json、protect.csv、protect_check、model_build | 実装確認。設定値と実保護数・実削減量を区別 |
| 異常の発生箇所を追う（後から追加） | layernorm_health/layernorm_lrp、score_health、invalid_score_batch、invalid_fitness | #012中心。記録だけで非有限値を修復・停止しない |
| 数値修復で異常を隠していないか | 修復前後統計、snapshots、execution_failed | #012/#015。SIGKILLはOSログが必要 |
| 観測コスト | timing、diag_seconds等 | 計時範囲外の監視もあるのでログなし速度と同一視しない |

## 保存の粒度

metadataは実行単位、スコアはバッチ単位、EAは全世代・試行単位。通常のLRP詳細は先頭2バッチが既定。追加LayerNorm健全性監視は全バッチだが、snapshotは別の共有保存上限がある。CSV統計・event・NPZ・tensor snapshotは粒度が違う。

## 追加の経緯

1. 最初に残差/Filter/bias/スコア集約/EA/実体整合性を観測するログを追加。
2. 10/4にNaN原因を追うLayerNorm・新規/累積スコア・異常画像の監視を追加。同時に数値修正した別実験も存在。
3. その後数値修正・停止・乱数固定・候補ソートを撤回。監視と集計だけを残す。
4. 10/13のEVO_IMNET選択肢追加はログではなく入力選択条件の変更。ログ専用の同条件比較と混ぜない。

公開済みコードに後からの監視実装がすべてコミットされているわけではない。本コミットは説明・目録・復元根拠を保存するもので、コードの公開状態を変えない。
