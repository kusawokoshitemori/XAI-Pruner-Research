"""
診断ログの共通基盤（観測専用）

方針（ログ追加で実験そのものを変えない）:
  - 乱数を一切引かない（random / np.random / torch の乱数状態に触れない）
  - 計算中の tensor を in-place で変更しない。記録は detach() したコピーから行う
  - 手法側の値（scores, candidates, masks, best）を書き換えない
  - 無効時（既定）は enabled() が False を返すだけで、手法の計算経路は変わらない
  - ログ処理に使った時間を別集計し、手法の計算時間と混ぜない
"""
import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # 解析スクリプトなど torch なしで import される場合
    torch = None


# 実測値と説明用の値を混ぜないための状態ラベル
STATUS_MEASURED = "measured"


class _State(object):
    def __init__(self):
        self.enabled = False
        self.root = None
        self.run_id = None
        self.rank = 0
        self.events = None
        self.tables = dict()
        self.seq = 0

        # LRP 内部ログを出すバッチ数の上限（重いため）
        self.lrp_batches = 0
        self.lrp_active = False
        self.batch_id = -1
        self.phase = None

        # 異常時 tensor 保存の残り回数
        self.snapshot_budget = 0
        self.near_zero_threshold = 1e-4
        self.flip_threshold = 1e-8
        self.max_quantile_elements = 1 << 22

        self.diag_seconds = 0.0
        self.t_start = None
        self.score_version = None


STATE = _State()


def enabled():
    return STATE.enabled


def lrp_active():
    return STATE.enabled and STATE.lrp_active


class timed(object):
    """ログ処理の所要時間を手法の時間と分けて集計する"""

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        STATE.diag_seconds += time.perf_counter() - self.t
        return False


# ----------------------------------------------------------------------------
# 値の変換
# ----------------------------------------------------------------------------
def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if torch is not None and isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Inf" if value > 0 else "-Inf"
    return value


def safe_div(a, b):
    """分母 0・非有限なら None（= NA）"""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return None
    if b == 0 or not math.isfinite(a) or not math.isfinite(b):
        return None
    return a / b


# ----------------------------------------------------------------------------
# 出力
# ----------------------------------------------------------------------------
class CsvTable(object):
    def __init__(self, path):
        self.path = path
        self.file = None
        self.writer = None
        self.fields = None

    def write(self, row):
        row = {k: _csv_cell(v) for k, v in row.items()}
        if self.writer is None:
            self.fields = list(row.keys())
            self.file = open(self.path, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields, restval="")
            self.writer.writeheader()
        # 未知の列が来たら黙って捨てずに例外にする（DictWriter の既定動作）
        self.writer.writerow(row)

    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()


def _csv_cell(v):
    v = to_jsonable(v)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def emit(event, **fields):
    if not STATE.enabled:
        return
    row = {
        "run_id": STATE.run_id,
        "rank": STATE.rank,
        "seq": STATE.seq,
        "time_ns": time.time_ns(),
        "phase": STATE.phase,
        "batch_id": STATE.batch_id,
        "event": event,
        "status": STATUS_MEASURED,
    }
    row.update(fields)
    STATE.seq += 1
    STATE.events.write(json.dumps(to_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")


def table(name, row):
    if not STATE.enabled:
        return
    t = STATE.tables.get(name)
    if t is None:
        t = CsvTable(os.path.join(STATE.root, name + ".csv"))
        STATE.tables[name] = t
    t.write(row)


def save_npz(name, **arrays):
    if not STATE.enabled:
        return
    np.savez_compressed(os.path.join(STATE.root, name), **arrays)


def save_snapshot(name, **tensors):
    """最初の NaN/Inf など、異常があった演算の tensor だけを保存する"""
    if not lrp_active() or STATE.snapshot_budget <= 0:
        return None
    STATE.snapshot_budget -= 1
    d = os.path.join(STATE.root, "snapshots")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "{:04d}_{}.pt".format(STATE.seq, name.replace("/", "_")))
    torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in tensors.items()}, path)
    return path


# ----------------------------------------------------------------------------
# 初期化・終了
# ----------------------------------------------------------------------------
def add_args(parser):
    g = parser.add_argument_group("diagnostics (観測ログ。指定しなければ何も出力しない)")
    g.add_argument("--diag_dir", default="", type=str,
                   help="診断ログの出力先。空なら無効")
    g.add_argument("--diag_lrp_batches", default=2, type=int,
                   help="LRP 内部（加算・Linear・Filter・Conv）の詳細ログを出す先頭バッチ数")
    g.add_argument("--diag_snapshots", default=5, type=int,
                   help="非有限値などが出た演算の tensor を保存する最大回数")
    g.add_argument("--diag_near_zero", default=1e-4, type=float,
                   help="近ゼロ分母の診断閾値 |z| < この値（誤計算の確定ラベルではない）")
    g.add_argument("--diag_flip_threshold", default=1e-8, type=float,
                   help="符号反転の閾値付き集計で、|値| がこれ以上のものだけを数える")
    return parser


def _git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=here, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def init(args, rank=0, script=None):
    """args.diag_dir が空なら何もしない"""
    if not getattr(args, "diag_dir", ""):
        return False

    run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + hashlib.sha1(
        json.dumps(to_jsonable(vars(args)), sort_keys=True).encode()).hexdigest()[:8]
    root = os.path.join(args.diag_dir, run_id, "rank{}".format(rank))
    os.makedirs(root, exist_ok=True)

    STATE.enabled = True
    STATE.root = root
    STATE.run_id = run_id
    STATE.rank = rank
    STATE.events = open(os.path.join(root, "events.jsonl"), "a", encoding="utf-8")
    STATE.lrp_batches = args.diag_lrp_batches
    STATE.snapshot_budget = args.diag_snapshots
    STATE.near_zero_threshold = args.diag_near_zero
    STATE.flip_threshold = args.diag_flip_threshold
    STATE.t_start = time.perf_counter()

    meta = {
        "run_id": run_id,
        "rank": rank,
        "script": script,
        "argv": sys.argv,
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", None),
        "cuda": (torch.version.cuda if torch is not None else None),
        "cudnn_benchmark": (torch.backends.cudnn.benchmark if torch is not None else None),
        "git": _git_info(),
        # 11.11: EA の親選択は Python の random を使うが、スクリプトは seed を設定していない
        "seeds": {
            "torch_manual_seed": getattr(args, "seed", None),
            "numpy_seed": getattr(args, "seed", None),
            "python_random_seeded_by_script": False,
            "python_random_state_sha1_at_init": _python_random_state_hash(),
        },
        # random.sample(set, k) は Python 3.11 以降 TypeError になる
        "random_sample_accepts_set": sys.version_info < (3, 11),
        "diag_thresholds": {
            "near_zero": STATE.near_zero_threshold,
            "flip": STATE.flip_threshold,
        },
    }
    with open(os.path.join(root, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(meta), f, ensure_ascii=False, indent=2)
    print("[diag] logging to {}".format(root))
    return True


def _python_random_state_hash():
    import random
    # getstate() は状態を読むだけで乱数を消費しない
    return hashlib.sha1(repr(random.getstate()).encode()).hexdigest()


def set_phase(phase):
    if STATE.enabled:
        STATE.phase = phase
        emit("phase_start", python_random_state_sha1=_python_random_state_hash())


def begin_batch(batch_id):
    if not STATE.enabled:
        return
    STATE.batch_id = batch_id
    STATE.lrp_active = batch_id < STATE.lrp_batches


def end_batch():
    if not STATE.enabled:
        return
    STATE.lrp_active = False
    STATE.events.flush()


def close():
    if not STATE.enabled:
        return
    total = time.perf_counter() - STATE.t_start
    emit("timing",
         wall_seconds_since_init=total,
         diag_seconds=STATE.diag_seconds,
         method_seconds_estimate=total - STATE.diag_seconds,
         note="diag_seconds は記録処理の時間。GPU 同期による待ちは手法側に混ざり得るので、"
              "純粋な計算時間はログ無効の実行で別途測ること")
    for t in STATE.tables.values():
        t.close()
    STATE.events.close()
    STATE.enabled = False


# ----------------------------------------------------------------------------
# tensor / 配列の要約
# ----------------------------------------------------------------------------
_QS = (0.0, 0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def tensor_summary(t, quantiles=True):
    """有限値のみの総和は finite_* と明記する（非有限を含む全体の総和とは呼ばない）"""
    if t is None:
        return None
    x = t.detach().reshape(-1)
    n = x.numel()
    if n == 0:
        return {"shape": list(t.shape), "numel": 0}
    x = x.to(torch.float64)
    fin = torch.isfinite(x)
    xf = torch.where(fin, x, torch.zeros_like(x))
    ax = xf.abs()
    stats = torch.stack([
        fin.sum().to(torch.float64),
        torch.isnan(x).sum().to(torch.float64),
        torch.isposinf(x).sum().to(torch.float64),
        torch.isneginf(x).sum().to(torch.float64),
        (x == 0).sum().to(torch.float64),
        xf.sum(),
        ax.sum(),
        ax.max(),
        (xf > 0).sum().to(torch.float64),
        (xf < 0).sum().to(torch.float64),
    ]).tolist()
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": n,
        "finite_count": int(stats[0]),
        "nan_count": int(stats[1]),
        "posinf_count": int(stats[2]),
        "neginf_count": int(stats[3]),
        "zero_count": int(stats[4]),
        "finite_signed_sum": stats[5],
        "finite_abs_sum": stats[6],
        "finite_max_abs": stats[7],
        "positive_count": int(stats[8]),
        "negative_count": int(stats[9]),
    }
    if quantiles and out["finite_count"] > 0:
        out["finite_abs_quantiles"] = quantile_dict(x[fin].abs())
    return out


def quantile_dict(v, qs=_QS):
    """大きい tensor は等間隔の間引きで分位数を計算する（乱数は使わない）"""
    v = v.detach().reshape(-1).to(torch.float64)
    if v.numel() == 0:
        return None
    sub = False
    if v.numel() > STATE.max_quantile_elements:
        step = int(math.ceil(v.numel() / STATE.max_quantile_elements))
        v = v[::step]
        sub = True
    q = torch.quantile(v, v.new_tensor(qs)).tolist()
    d = {"q{:g}".format(k * 100): val for k, val in zip(qs, q)}
    d["mean"] = v.mean().item()
    if sub:
        d["strided_subsample"] = True
    return d


def per_sample_quantiles(v):
    """サンプルごとの値ベクトルを、サンプル間の分位数に要約する（全体統計がサンプル差を隠さないように）"""
    v = v.detach().reshape(-1).to(torch.float64)
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return None
    return quantile_dict(v, qs=(0.0, 0.25, 0.5, 0.75, 1.0))


def array_summary(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    fin = np.isfinite(a)
    out = {
        "count": int(a.size),
        "finite_count": int(fin.sum()),
        "nan_count": int(np.isnan(a).sum()),
        "posinf_count": int(np.isposinf(a).sum()),
        "neginf_count": int(np.isneginf(a).sum()),
    }
    f = a[fin]
    if f.size:
        out.update({
            "sum": float(f.sum()),
            "mean": float(f.mean()),
            "min": float(f.min()),
            "q01": float(np.quantile(f, 0.01)),
            "q10": float(np.quantile(f, 0.10)),
            "median": float(np.median(f)),
            "q90": float(np.quantile(f, 0.90)),
            "q99": float(np.quantile(f, 0.99)),
            "max": float(f.max()),
        })
    return out


# ----------------------------------------------------------------------------
# 識別子
# ----------------------------------------------------------------------------
def candidate_id(candidate):
    """削除率ベクトルの識別子。丸めずに float の16進表現を使う"""
    payload = [float(v).hex() for v in candidate]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def mask_id(masks):
    """実際の削除対象（二値マスク）の識別子。非二値なら例外にして隠さない"""
    h = hashlib.sha256()

    def visit(obj, path):
        if isinstance(obj, dict):
            for key in sorted(obj):
                visit(obj[key], path + [str(key)])
            return
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                visit(item, path + [str(i)])
            return
        arr = obj.detach().cpu().numpy() if (torch is not None and torch.is_tensor(obj)) else np.asarray(obj)
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError("非二値マスクです: {}".format(path))
        b = np.ascontiguousarray(arr.astype(np.uint8))
        h.update(json.dumps({"path": path, "shape": list(b.shape)}, sort_keys=True).encode())
        h.update(b.tobytes())

    visit(masks, [])
    return h.hexdigest()[:16]


def counts_mask_id(kind, counts):
    """
    スコア順位が固定のとき、削除数の組からマスクは一意に決まる。
    そのため EA の各試行では、マスク本体を作らずに (score_version, 削除数) から mask_id を作る。
    最終マスクについては mask_id() で本体からも計算し、一致を確認する。
    """
    payload = [kind, STATE.score_version] + [int(c) for c in counts]
    return "c" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:15]


def arrays_version(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:12]


def spearman(a, b):
    """scipy なしの Spearman（平均順位で tie を扱う）"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    # tie を平均順位に
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / cnt)[inv]
