"""
関連度スコアの集約に関する観測ログ（11.8, 11.9, 8.2, 8.4, 11.6）
"""
import copy
import hashlib

import numpy as np

from .core import (STATE, emit, table, save_npz, array_summary, arrays_version, spearman, safe_div, timed)

try:
    import torch
except ImportError:
    torch = None


# ----------------------------------------------------------------------------
# 11.9 : 初期関連度（損失ではなく予測クラスの softmax 確率）
# ----------------------------------------------------------------------------
def initial_relevance(batch_id, samples, outputs, targets, relevance, momentum):
    if not STATE.enabled:
        return
    with timed():
        out = outputs.detach().float()
        prob = torch.softmax(out, dim=1)
        conf, pred = prob.max(dim=1)
        rel = relevance.detach().float()
        rel_class = rel.argmax(dim=1)
        rel_amount = rel.sum(dim=1)
        tgt = targets.detach()
        correct = (pred == tgt)
        x = samples.detach().cpu()

        pred_l, tgt_l = pred.tolist(), tgt.tolist()
        conf_l, rc_l, ra_l, cor_l = conf.tolist(), rel_class.tolist(), rel_amount.tolist(), correct.tolist()
        for i in range(x.shape[0]):
            table("initial_relevance", {
                "batch_id": batch_id,
                "position_in_batch": i,
                # データローダーは index を返さないため、前処理後の画像 tensor のハッシュをサンプル識別に使う
                "sample_hash": hashlib.sha1(x[i].contiguous().numpy().tobytes()).hexdigest()[:16],
                "true_label": tgt_l[i],
                "predicted_class": pred_l[i],
                "initial_relevance_class": rc_l[i],
                "initial_relevance_amount": ra_l[i],
                "prediction_confidence": conf_l[i],
                "baseline_correct": bool(cor_l[i]),
            })
        table("score_batches", {
            "batch_id": batch_id,
            "batch_size": x.shape[0],
            "momentum": momentum,
            "baseline_correct_count": int(correct.sum().item()),
            "mean_confidence": float(conf.mean().item()),
            "initial_relevance_total": float(rel_amount.sum().item()),
            "relevance_class_equals_true_label": int((rel_class == tgt).sum().item()),
        })


# ----------------------------------------------------------------------------
# 8.2 : ViT の集約要素数・非ゼロ要素数（バッチ間の単純平均）
# ----------------------------------------------------------------------------
_vit_nonzero = {"batches": 0}


def vit_relevance(relevance):
    if not STATE.enabled:
        return
    with timed():
        acc = _vit_nonzero
        heads = [(r != 0).sum(dim=(2, 3)).double().mean(dim=0).cpu().numpy() for r in relevance["head"]]
        hidden = [(r != 0).sum(dim=1).double().mean(dim=0).cpu().numpy() for r in relevance["hidden"]]
        dim = (relevance["dim"] != 0).sum(dim=1).double().mean(dim=0).cpu().numpy()
        if acc["batches"] == 0:
            acc["head"] = np.asarray(heads)
            acc["hidden"] = np.asarray(hidden)
            acc["dim"] = dim
            r0 = relevance["head"][0]
            acc["n_head"] = int(r0.shape[2] * r0.shape[3])
            acc["n_hidden"] = int(relevance["hidden"][0].shape[1])
            acc["n_dim"] = int(relevance["dim"].shape[1])
            acc["raw_shape"] = {"head": list(r0.shape), "hidden": list(relevance["hidden"][0].shape),
                                "dim": list(relevance["dim"].shape)}
        else:
            acc["head"] = acc["head"] + np.asarray(heads)
            acc["hidden"] = acc["hidden"] + np.asarray(hidden)
            acc["dim"] = acc["dim"] + dim
        acc["batches"] += 1


# ----------------------------------------------------------------------------
# 11.8 : momentum 集計（後ろのバッチほど強く残る）と、診断用の単純平均の併記
# ----------------------------------------------------------------------------
_plain = {"sum": None, "batches": 0}


def _flatten(scores):
    """スコア dict を {名前: 1次元配列} に（ViT / ResNet / VGG 共通）"""
    out = {}
    for key, value in scores.items():
        if isinstance(value, np.ndarray) and value.dtype != object:
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 2:
                for i in range(arr.shape[0]):
                    out["{}/{}".format(key, i)] = arr[i].reshape(-1)
            else:
                out[key] = arr.reshape(-1)
        else:
            for i, v in enumerate(value):
                out["{}/{}".format(key, i)] = np.asarray(v, dtype=np.float64).reshape(-1)
    return out


def batch_scores(batch_id, new_scores, accumulated, momentum):
    if not STATE.enabled:
        return
    with timed():
        flat_new = _flatten(new_scores)
        flat_acc = _flatten(accumulated)
        if _plain["sum"] is None:
            _plain["sum"] = {k: v.copy() for k, v in flat_new.items()}
        else:
            for k, v in flat_new.items():
                _plain["sum"][k] = _plain["sum"][k] + v
        _plain["batches"] += 1
        for k in flat_new:
            row = {"batch_id": batch_id, "group": k, "momentum": momentum}
            for prefix, arr in (("batch", flat_new[k]), ("accumulated", flat_acc[k])):
                s = array_summary(arr)
                row.update({"{}_{}".format(prefix, n): s.get(n) for n in ("sum", "mean", "median", "max", "nan_count", "posinf_count")})
            table("score_accumulation", row)


def finalize(scores, momentum):
    """momentum 集計と単純平均の順位を比較する。手法のスコアは変更しない"""
    if not STATE.enabled or _plain["sum"] is None:
        return
    with timed():
        flat = _flatten(scores)
        T = _plain["batches"]
        mean = {k: v / T for k, v in _plain["sum"].items()}
        groups = {}
        for k in flat:
            g = k.split("/")[0]
            groups.setdefault(g, []).append(k)
        comp = {}
        for g, keys in groups.items():
            a = np.concatenate([flat[k] for k in keys])
            b = np.concatenate([mean[k] for k in keys])
            overlaps = {}
            for q in (0.1, 0.3, 0.5):
                n = int(a.size * q)
                if n:
                    sa = set(np.argsort(a, kind="mergesort")[:n].tolist())
                    sb = set(np.argsort(b, kind="mergesort")[:n].tolist())
                    overlaps["lowest_{:g}pct_overlap".format(q * 100)] = len(sa & sb) / n
            comp[g] = {"spearman_momentum_vs_plain_mean": spearman(a, b), **overlaps}
        emit("score_aggregation_comparison",
             batches=T, momentum=momentum,
             final_weight_of_first_batch=momentum ** (T - 1) if T else None,
             note="現行は scores = momentum*scores + new。重み m^(T-t) で後ろのバッチほど強い",
             by_group=comp)
        save_npz("scores_momentum.npz", **{k.replace("/", "__"): v for k, v in flat.items()})
        save_npz("scores_plain_mean.npz", **{k.replace("/", "__"): v for k, v in mean.items()})


# ----------------------------------------------------------------------------
# 8.4 / 11.6 : protect() の前後
# ----------------------------------------------------------------------------
def copy_scores(scores):
    return copy.deepcopy(scores)


def protect_report(kind, controller, before, after):
    if not STATE.enabled:
        return
    with timed():
        fb, fa = _flatten(before), _flatten(after)
        for k in fb:
            table("protect", {
                "group": k,
                "count": fb[k].size,
                "requested_protect_rate": controller.protect_percent,
                "requested_protect_count": _requested_protect(kind, controller, k, fb[k].size),
                "actual_protected_count": int(np.sum(fa[k] != fb[k])),
                "finite_before": int(np.isfinite(fb[k]).sum()),
                "finite_after": int(np.isfinite(fa[k]).sum()),
                "before": array_summary(fb[k]),
                "after": array_summary(fa[k]),
            })
        if kind == "vit":
            req = int(np.ceil(controller.protect_percent * controller.num_heads_per_layer))
            emit("protect_check",
                 kind=kind, requested_protect_count_per_layer=req,
                 note=("protect_count=0 だと sorted_head[i][-0:] が全要素になり全 Head が Inf になる"
                       if req == 0 else None),
                 head_inf_after=int(np.isinf(np.asarray(after["head"])).sum()),
                 head_total=int(np.asarray(after["head"]).size))

        STATE.score_version = arrays_version([fa[k] for k in sorted(fa)])
        emit("score_version", score_version=STATE.score_version,
             note="protect() 後のスコア。EA の mask_id はこの版のスコア順位に対して定義する")
        save_npz("scores_after_protect.npz", **{k.replace("/", "__"): v for k, v in fa.items()})


def _requested_protect(kind, c, key, n):
    if kind == "vit":
        return int(np.ceil(c.protect_percent * c.num_heads_per_layer)) if key.startswith("head") else 0
    if kind == "resnet":
        # 各ブロック最後の配列は 0.5 倍の保護率（protect() の実装どおり）
        return None
    return None


# ----------------------------------------------------------------------------
# 8.2 : 構造単位の表（ViT）
# ----------------------------------------------------------------------------
def vit_component_table(controller, model, scores_before_protect):
    if not STATE.enabled:
        return
    with timed():
        C = controller.embed_dim
        H = controller.num_heads_per_layer
        Dh = C // H
        hid = controller.hidden_dim_per_layer
        depth = controller.depth
        acc = _vit_nonzero
        nb = max(acc["batches"], 1)
        p = model.patch_size if isinstance(model.patch_size, int) else model.patch_size[0]
        n_tok = model.num_patches + 1
        # 単独削除（他の構造はすべて残す条件）で減るパラメータ数。deploy.get_pruned_model_vit の添字規則に合わせた定義
        g_head = 4 * C * Dh + 3 * Dh
        g_hidden = 2 * C + 1
        g_dim = (3 * p * p + 1) + 1 + n_tok + depth * (2 + 3 * C + C + 1 + 2 + 2 * hid + 1) + 2 + model.num_classes
        emit("component_size_definition",
             parameter_count_definition="単独削除で減るパラメータ数（他の構造は残す）。Head と Dim は重複するので和は同時削除の削減量ではない",
             g_head=g_head, g_hidden=g_hidden, g_dim=g_dim,
             n_head=acc.get("n_head"), n_hidden=acc.get("n_hidden"), n_dim=acc.get("n_dim"),
             raw_shape=acc.get("raw_shape"),
             sum_axes={"head": "tokens x head_dim (dims 2,3)", "hidden": "tokens (dim 1)", "dim": "tokens (dim 1)"})

        def rows(ctype, arr2d, nonzero2d, n, g):
            for l in range(arr2d.shape[0]):
                for j in range(arr2d.shape[1]):
                    s = float(arr2d[l, j])
                    table("component_scores", {
                        "component_type": ctype, "layer_id": l, "component_id": j,
                        "score_sum_abs": s,
                        "n_aggregated": n,
                        "mean_nonzero_element_count": (float(nonzero2d[l, j]) / nb) if nonzero2d is not None else None,
                        "parameter_count_single_deletion": g,
                        "score_per_aggregated_element": safe_div(s, n),
                        "score_per_parameter": safe_div(s, g),
                    })

        rows("head", np.asarray(scores_before_protect["head"]), acc.get("head"), acc.get("n_head"), g_head)
        rows("hidden", np.asarray(scores_before_protect["hidden"]), acc.get("hidden"), acc.get("n_hidden"), g_hidden)
        d = np.asarray(scores_before_protect["dim"]).reshape(1, -1)
        nz = acc.get("dim")
        rows("dim", d, None if nz is None else np.asarray(nz).reshape(1, -1), acc.get("n_dim"), g_dim)
