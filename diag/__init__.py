"""
観測専用の診断ログ（ログ出したいもの.txt の層A＋第0段階の整合性ログ）。

--diag_dir を指定したときだけ有効。無効時は手法の計算経路を一切変えない。
"""
from .core import (STATE, enabled, lrp_active, init, close, add_args, emit, table,
                   set_phase, begin_batch, end_batch)
from . import relevance, scores, ea
