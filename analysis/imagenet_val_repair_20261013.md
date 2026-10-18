# ImageNet検証セットの振り分け修復（2026-10-13）

対象: `/mnt/data1/imagenet/val`。移動前はクラス内36,381枚、直下13,619枚。検証画像名00000001〜00050000の全件が存在し、欠番・重複なし。

クラス対応表: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

取得内容SHA-256: `dc5a6a8ac37f33adfb6e949b7b2fa42fd84600f38c58089e60a674f2a289d67d`。シェルスクリプトは実行せず、mv行から対応表を読み取った。既存36,381枚の全所属先が対応表と一致し、対応表が1000クラス×50枚であることを確認後、未分類画像のみ移動した。既存画像を上書きせず、元のtarも保持。

移動後はtorchvision.datasets.ImageFolderで50,000画像・1,000クラス・各50枚を確認。移動処理記録はローカル `/tmp/xai_val_moves.jsonl`（一時ファイル）。

以前の結果は36,381枚の検証セットによる値として保持。今回以降の50,000枚の評価値と同一条件として混ぜない。valを関連度計算に用いていた以前の枝刈り実験も、データが変わるため再実行では条件差を記録する。train/subImageNetと訓練コードはこの修復では変更していない。
