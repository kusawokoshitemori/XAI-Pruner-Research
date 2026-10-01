"""
EA（交叉・突然変異・ランダム補充・Top-k）と、候補 → マスク → 削除モデルの整合性の観測ログ
（10.x, 11.1-11.7, 12, 8.5）

手法側の乱数・候補集合・スコア・マスクには触れない。is_legal() / get_complexity() など
副作用のない関数だけを再利用する。
"""
import copy
import time

import numpy as np

from .core import (STATE, emit, table, candidate_id, mask_id, counts_mask_id, safe_div,
                   spearman, timed)


class _Tracker(object):
    def __init__(self, controller, kind):
        self.c = controller
        self.kind = kind
        self.epoch = -1
        self.parent_pool = set()
        self.parent_masks = set()
        self.cache = {}
        self.attempt_seq = 0
        self.op_masks = set()
        self.select_info = None
        self.random_rows = []
        self._prepare_fitness()

    # --- Fitness の内訳（calculate_fitness と同じ式・同じ切り出し）
    def _prepare_fitness(self):
        c = self.c
        if self.kind == "vit":
            self.sorted_head = np.sort(np.abs(np.asarray([s for layer in c.scores["head"] for s in layer])))
            self.sorted_hidden = np.sort(np.abs(np.asarray([s for layer in c.scores["hidden"] for s in layer])))
            self.sorted_dim = np.sort(np.abs(np.asarray(c.scores["dim"])))
            self.totals = (c.embed_dim, c.hidden_dim, c.num_heads)
        else:
            self.sorted_blocks = [np.sort(np.concatenate(v)) for v in c.scores.values()]
            self.totals = tuple(int(x) for x in c.dim_per_block)

    def counts(self, cand):
        c = self.c
        if self.kind == "vit":
            # calculate_fitness / generate_masks と同じ int() 切り捨て
            return (int(c.embed_dim * cand[0]), int(c.hidden_dim * cand[1]), int(c.num_heads * cand[2]))
        return tuple(int(x) for x in np.round(c.dim_per_block * np.asarray(cand)).astype(int))

    def info(self, cand):
        key = tuple(float(x) for x in cand)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        c = self.c
        r = np.asarray(cand, dtype=np.float64)
        cnt = self.counts(cand)
        d = {
            "candidate_id": candidate_id(cand),
            "mask_id": counts_mask_id(self.kind, cnt),
            "rates": [float(x) for x in r],
            "counts": list(cnt),
            "min_rate": float(r.min()),
            "max_rate": float(r.max()),
            "negative_gene_count": int((r < 0).sum()),
            "zero_gene_count": int((r == 0).sum()),
            "greater_than_one_gene_count": int((r > 1).sum()),
            # 実装の判定（ResNet は np.all(rate) > 0 なので負の値を検査できていない）
            "implementation_legal": bool(c.is_legal(np.asarray(cand))),
            # 意味上の有効性: 0 <= 削除数 <= 構造数
            "diagnostic_bounds_ok": bool(all(0 <= n <= t for n, t in zip(cnt, self.totals))),
        }
        if self.kind == "vit":
            flops = c.model.get_complexity(dim_rate=cand[0], hidden_rate=cand[1], head_rate=cand[2])
            d["constraint_value"] = float(1 - flops / c.original_flops)
            d["constraint_kind"] = "flops_reduction(get_complexity)"
            f_dim = float(np.sum(self.sorted_dim[:cnt[0]]))
            f_hidden = float(np.sum(self.sorted_hidden[:cnt[1]]))
            f_head = float(np.sum(self.sorted_head[:cnt[2]]))
            parts = {"F_dim": f_dim, "F_hidden": f_hidden, "F_head": f_head}
        else:
            d["constraint_value"] = float(c.calculate_pruned_percentage(cand))
            d["constraint_kind"] = "pruned_channels/original_channels (not FLOPs)"
            parts = {"F_block{}".format(i): float(np.sum(s[:n])) for i, (s, n) in
                     enumerate(zip(self.sorted_blocks, cnt))}
        total = float(sum(parts.values()))
        d["fitness_parts"] = parts
        d["F_total_recomputed"] = total
        d["fitness_part_shares"] = {k: safe_div(v, total) for k, v in parts.items()}
        d["constraint_delta"] = d["constraint_value"] - c.percentage
        self.cache[key] = d
        return d


_T = None


def start(controller, kind):
    global _T
    if not STATE.enabled:
        return
    with timed():
        _T = _Tracker(controller, kind)
        c = controller
        emit("ea_config", kind=kind, population=c.population, epochs=c.epochs,
             target=c.percentage, epsilon=c.epsilon, protect=c.protect_percent,
             note_top_k="top_k は割合ではなく絶対個数",
             note_constraint=("ViT: get_complexity による FLOPs 削減率" if kind == "vit" else
                              "ResNet: 削除チャネル数 / 元チャネル数（実測 FLOPs ではない）"))


# ----------------------------------------------------------------------------
# 世代
# ----------------------------------------------------------------------------
def begin_generation(epoch, previous, parent_pool, random_attempts):
    if _T is None:
        return
    with timed():
        _T.epoch = epoch
        _T.previous = set(previous)
        _T.parent_pool = set(parent_pool)
        _T.parent_masks = {_T.info(p)["mask_id"] for p in parent_pool}
        _T.random_attempts = random_attempts
        for row in _T.random_rows:
            row["epoch"] = epoch
            table("random_generator", row)
        _T.random_rows = []


def random_candidate(cand, added, internals=None):
    """世代開始時のランダム補充。返却値は追加前に is_legal() で再検査されていない"""
    if _T is None:
        return
    with timed():
        inf = _T.info(cand)
        row = {
            "epoch": None,
            "added_to_population": added,
            "candidate_id": inf["candidate_id"],
            "mask_id": inf["mask_id"],
            "returned_rates": inf["rates"],
            "returned_implementation_legal": inf["implementation_legal"],
            "returned_bounds_ok": inf["diagnostic_bounds_ok"],
            "constraint_value": inf["constraint_value"],
            "constraint_delta": inf["constraint_delta"],
        }
        for k in ("initial_rates", "scaling_iterations", "binary_search_iterations",
                  "termination_reason", "last_mid_step"):
            row[k] = (internals or {}).get(k)
        _T.random_rows.append(row)


def begin_operator():
    if _T is not None:
        _T.op_masks = set()


def attempt(op, attempt_id, requested, max_iters, parents, raw_child, final_child,
            legal_before, revised, added, crossover_mask=None, decisions=None, deltas=None):
    if _T is None:
        return
    with timed():
        raw_inf = _T.info(raw_child)
        fin = None if final_child is None else tuple(final_child)
        fin_inf = None if fin is None else _T.info(fin)
        p_inf = [_T.info(p) for p in parents]

        row = {
            "epoch": _T.epoch, "operator": op, "attempt_id": attempt_id,
            "requested": requested, "max_iters": max_iters,
            "source_parent_ids": [p["candidate_id"] for p in p_inf],
            "source_parent_mask_ids": [p["mask_id"] for p in p_inf],
            "raw_candidate": raw_inf["rates"],
            "raw_candidate_id": raw_inf["candidate_id"],
            "legal_before_revise": bool(legal_before),
            "revised": bool(revised),
            "revised_candidate": fin_inf["rates"] if (revised and fin_inf) else None,
            "legal_after_revise": (fin_inf["implementation_legal"] if (revised and fin_inf) else None),
            "final_candidate_id": fin_inf["candidate_id"] if fin_inf else None,
            "final_mask_id": fin_inf["mask_id"] if fin_inf else None,
            "final_bounds_ok": fin_inf["diagnostic_bounds_ok"] if fin_inf else None,
            "final_min_rate": fin_inf["min_rate"] if fin_inf else None,
            "final_max_rate": fin_inf["max_rate"] if fin_inf else None,
            "constraint_value": fin_inf["constraint_value"] if fin_inf else raw_inf["constraint_value"],
        }
        # 交叉・突然変異で列を揃える（該当しない列は空欄）
        for k in ("crossover_mask", "parent_gene_difference_count", "parent_mask_differs",
                  "child_same_as_parent1", "child_same_as_parent2", "child_mask_same_as_parent1",
                  "child_mask_same_as_parent2", "mutation_decision_mask", "mutation_deltas", "mutated_gene_count"):
            row[k] = None
        if op == "crossover":
            p1, p2 = parents
            row.update({
                "crossover_mask": [int(m) for m in crossover_mask],
                "parent_gene_difference_count": int(sum(float(a) != float(b) for a, b in zip(p1, p2))),
                "parent_mask_differs": p_inf[0]["mask_id"] != p_inf[1]["mask_id"],
                "child_same_as_parent1": (tuple(raw_child) == tuple(p1)),
                "child_same_as_parent2": (tuple(raw_child) == tuple(p2)),
                "child_mask_same_as_parent1": raw_inf["mask_id"] == p_inf[0]["mask_id"],
                "child_mask_same_as_parent2": raw_inf["mask_id"] == p_inf[1]["mask_id"],
            })
            same_src = row["child_same_as_parent1"] or row["child_same_as_parent2"]
        else:
            row.update({
                "mutation_decision_mask": [int(x) for x in decisions],
                "mutation_deltas": deltas,
                "mutated_gene_count": int(sum(decisions)),
                "child_mask_same_as_parent1": raw_inf["mask_id"] == p_inf[0]["mask_id"],
            })
            same_src = tuple(raw_child) == tuple(parents[0])
        row["same_as_source_parent"] = bool(same_src)

        in_pool = fin is not None and fin in _T.parent_pool
        mask_in_pool = fin_inf is not None and fin_inf["mask_id"] in _T.parent_masks
        novel_mask = bool(added and fin_inf and not mask_in_pool and fin_inf["mask_id"] not in _T.op_masks)
        if added and fin_inf:
            _T.op_masks.add(fin_inf["mask_id"])
        row.update({
            "already_in_parent_pool": bool(in_pool),
            "mask_already_in_parent_pool": bool(mask_in_pool),
            # 演算子内 kids 集合にすでにあったために増えなかった
            "already_in_operator_kids": bool(fin is not None and not added and (legal_before or revised)),
            "added_to_operator_kids": bool(added),
            "novel_genotype_added": bool(added and not in_pool),
            "novel_mask_added": novel_mask,
        })
        table("ea_attempts", row)


def operator_end(op, kids, iters, requested, max_iters, seconds):
    if _T is None:
        return
    with timed():
        kids = set(kids)
        masks = {_T.info(k)["mask_id"] for k in kids}
        overlap = kids & _T.parent_pool
        mask_overlap = masks & _T.parent_masks
        row = {
            "epoch": _T.epoch, "operator": op, "requested": requested, "max_iters": max_iters,
            "attempts": iters,
            "stop_reason": "requested_kids_reached" if len(kids) >= requested else "max_iters_reached",
            "nominal_target_reached": len(kids) >= requested,
            "returned_unique_genotypes": len(kids),
            "parent_overlap_unique_genotypes": len(overlap),
            "novel_unique_genotypes": len(kids - _T.parent_pool),
            "returned_unique_masks": len(masks),
            "parent_overlap_unique_masks": len(mask_overlap),
            "novel_unique_masks": len(masks - _T.parent_masks),
            "returned_illegal_or_out_of_bounds": sum(
                1 for k in kids if not (_T.info(k)["implementation_legal"] and _T.info(k)["diagnostic_bounds_ok"])),
            "operator_seconds_incl_diag": seconds,
        }
        table("ea_operator_calls", row)


def select(candidates, fitness, sorted_idx, top_k):
    """select() 内で、手法が計算した fitness と並びをそのまま受け取る"""
    if _T is None:
        return
    with timed():
        fitness = np.asarray(fitness, dtype=np.float64)
        rank = np.empty(len(candidates), dtype=int)
        full = np.argsort(fitness)
        rank[full] = np.arange(len(candidates))
        sel_idx = set(int(i) for i in sorted_idx)
        boundary = None
        if 0 < len(sorted_idx) < len(candidates):
            v = fitness[sorted_idx[-1]]
            tied = np.where(fitness == v)[0]
            boundary = {"value": float(v), "tie_count": int(tied.size),
                        "tied_selected": int(sum(int(i) in sel_idx for i in tied)),
                        "tied_not_selected": int(sum(int(i) not in sel_idx for i in tied))}
        _T.select_info = {"candidates": list(candidates), "fitness": fitness, "rank": rank,
                          "selected_idx": sel_idx, "boundary": boundary, "top_k": top_k}


def _source_metrics(source, selected):
    s = len(source & selected)
    return {"survived": s, "generated": len(source),
            "survival_rate": safe_div(s, len(source)), "selected_share": safe_div(s, len(selected))}


def generation(epoch, previous, parent_pool, cross_kids, mutation_kids, selected, top_k):
    if _T is None or _T.select_info is None:
        return
    with timed():
        previous, parent_pool = set(previous), set(parent_pool)
        C, M, S = set(cross_kids), set(mutation_kids), set(selected)
        pool = parent_pool | C | M
        random_added = parent_pool - previous
        cross_novel, mut_novel = C - parent_pool, M - parent_pool
        cross_only, mut_only, both = cross_novel - mut_novel, mut_novel - cross_novel, cross_novel & mut_novel

        si = _T.select_info
        cands, fit, rank = si["candidates"], si["fitness"], si["rank"]

        # 候補ごとの行（候補評価ログ）と、マスク単位の生成源
        mask_labels, mask_selected = {}, set()
        max_diff = 0.0
        comp_rows = []
        for i, cand in enumerate(cands):
            inf = _T.info(cand)
            labels = [n for n, s in (("previous", previous), ("random", random_added),
                                     ("crossover", C), ("mutation", M)) if cand in s]
            mask_labels.setdefault(inf["mask_id"], set()).update(labels)
            is_sel = i in si["selected_idx"]
            if is_sel:
                mask_selected.add(inf["mask_id"])
            if np.isfinite(fit[i]) and np.isfinite(inf["F_total_recomputed"]):
                max_diff = max(max_diff, abs(fit[i] - inf["F_total_recomputed"]))
            comp_rows.append(inf)
            table("candidate_evaluations", {
                "epoch": epoch, "candidate_id": inf["candidate_id"], "mask_id": inf["mask_id"],
                "score_version": STATE.score_version, "sources": labels,
                "rates": inf["rates"], "counts": inf["counts"], "totals": list(_T.totals),
                "fitness": float(fit[i]), "F_total_recomputed": inf["F_total_recomputed"],
                "fitness_parts": inf["fitness_parts"], "fitness_part_shares": inf["fitness_part_shares"],
                "constraint_kind": inf["constraint_kind"], "constraint_value": inf["constraint_value"],
                "implementation_legal": inf["implementation_legal"],
                "diagnostic_bounds_ok": inf["diagnostic_bounds_ok"],
                "negative_gene_count": inf["negative_gene_count"],
                "greater_than_one_gene_count": inf["greater_than_one_gene_count"],
                "rank": int(rank[i]), "selected": is_sel,
            })

        prev_masks = {_T.info(p)["mask_id"] for p in previous}
        mask_cat = {"existing": set(), "random_only": set(), "crossover_only": set(),
                    "mutation_only": set(), "multiple_ops": set()}
        for m, labels in mask_labels.items():
            if m in prev_masks:
                mask_cat["existing"].add(m)
                continue
            new_labels = labels - {"previous"}
            if len(new_labels) == 1:
                mask_cat[next(iter(new_labels)) + "_only"].add(m)
            else:
                mask_cat["multiple_ops"].add(m)

        # 種類別 Fitness 順位が総和順位をどれだけ決めているか（8.5）
        part_rank_corr = {}
        if comp_rows:
            tot = np.array([r["F_total_recomputed"] for r in comp_rows])
            for k in comp_rows[0]["fitness_parts"]:
                part_rank_corr[k] = spearman([r["fitness_parts"][k] for r in comp_rows], tot)

        finite = np.isfinite(fit)
        emit("ea_generation",
             epoch=epoch, requested_top_k=top_k,
             random_attempts=_T.random_attempts,
             pool_genotype_count=len(pool), selected_genotype_count=len(S),
             selected_genotype_fraction=safe_div(len(S), len(pool)),
             pool_unique_mask_count=len(mask_labels), selected_unique_mask_count=len(mask_selected),
             selected_mask_fraction=safe_div(len(mask_selected), len(mask_labels)),
             selected_duplicate_mask_count=len(S) - len(mask_selected),
             random_added_count=len(random_added),
             cross_returned=len(C), cross_parent_overlap=len(C & parent_pool), cross_novel=len(cross_novel),
             mutation_returned=len(M), mutation_parent_overlap=len(M & parent_pool), mutation_novel=len(mut_novel),
             sources={"previous": _source_metrics(previous, S), "random": _source_metrics(random_added, S),
                      "cross_only": _source_metrics(cross_only, S),
                      "mutation_only": _source_metrics(mut_only, S), "both_ops": _source_metrics(both, S)},
             mask_sources={k: _source_metrics(v, mask_selected) for k, v in mask_cat.items()},
             note_sources="生成源は直近の生成方法。祖先全体の寄与ではない",
             fitness_unique_count=int(np.unique(fit[finite]).size),
             fitness_nonfinite_count=int((~finite).sum()),
             topk_boundary_tie=si["boundary"],
             fitness_recompute_max_abs_diff=max_diff,
             pool_illegal_or_out_of_bounds=sum(1 for r in comp_rows if not (r["implementation_legal"] and r["diagnostic_bounds_ok"])),
             selected_illegal_or_out_of_bounds=sum(
                 1 for i, r in enumerate(comp_rows) if i in si["selected_idx"] and
                 not (r["implementation_legal"] and r["diagnostic_bounds_ok"])),
             best_candidate_id=_T.info(_T.c.best)["candidate_id"],
             best_fitness=float(fit[np.argsort(fit)[0]]) if len(fit) else None,
             spearman_fitness_part_vs_total=part_rank_corr)
        _T.select_info = None
        STATE.events.flush()


# ----------------------------------------------------------------------------
# 12 : 候補 → 想定削除集合 → 最終マスク の整合性
# ----------------------------------------------------------------------------
def final_vit(controller):
    if _T is None:
        return
    with timed():
        c = controller
        inf = _T.info(c.best)
        m = c.masks
        head_flat = np.asarray([s for layer in c.scores["head"] for s in layer])
        hidden_flat = np.asarray([s for layer in c.scores["hidden"] for s in layer])
        dim_flat = np.asarray(c.scores["dim"])
        deleted = {"dim": int((m["embed_dim"] == 0).sum()), "hidden": int((m["hidden_dim"] == 0).sum()),
                   "head": int((m["heads"] == 0).sum())}
        f_final = (float(np.abs(dim_flat[m["embed_dim"].reshape(-1) == 0]).sum()) +
                   float(np.abs(hidden_flat[m["hidden_dim"].reshape(-1) == 0]).sum()) +
                   float(np.abs(head_flat[m["heads"].reshape(-1) == 0]).sum()))
        method_f = float(c.calculate_fitness([c.best])[0])
        emit("final_consistency",
             kind="vit", candidate_id=inf["candidate_id"], rates=inf["rates"],
             implementation_legal=inf["implementation_legal"], diagnostic_bounds_ok=inf["diagnostic_bounds_ok"],
             candidate_requested_counts={"dim": inf["counts"][0], "hidden": inf["counts"][1], "head": inf["counts"][2]},
             mask_deleted_counts=deleted,
             counts_match=(deleted == {"dim": inf["counts"][0], "hidden": inf["counts"][1], "head": inf["counts"][2]}),
             fitness_score=method_f, fitness_recomputed_from_final_mask=f_final,
             fitness_relative_diff=safe_div(abs(method_f - f_final), abs(method_f)),
             fitness_mask_consistent=bool(np.isclose(method_f, f_final, rtol=1e-5, atol=0.0)),
             candidate_mask_id=inf["mask_id"],
             final_mask_id_from_deleted_counts=counts_mask_id("vit", (deleted["dim"], deleted["hidden"], deleted["head"])),
             final_mask_hash=mask_id(m),
             note_mask_id="candidate_mask_id と final_mask_id_from_deleted_counts は同じ規則（削除数）。final_mask_hash はマスク本体のハッシュ",
             deleted_heads_per_layer=(m["heads"] == 0).sum(axis=1).tolist(),
             deleted_hidden_per_layer=(m["hidden_dim"] == 0).sum(axis=1).tolist(),
             constraint_value=inf["constraint_value"])


def final_resnet(controller, masks_before_alignment):
    if _T is None:
        return
    with timed():
        c = controller
        inf = _T.info(c.best)
        blocks = list(c.scores.values())

        def deleted_and_fitness(masks):
            dels, f = [], 0.0
            for value, mk in zip(blocks, masks):
                s = np.concatenate(value)
                k = np.concatenate([np.asarray(x).reshape(-1) for x in mk]).astype(bool)
                dels.append(int((~k).sum()))
                f += float(s[~k].sum())
            return dels, f

        d0, f0 = deleted_and_fitness(masks_before_alignment)
        d1, f1 = deleted_and_fitness(c.masks)
        changed = sum(int((np.asarray(a).astype(bool) != np.asarray(b).astype(bool)).sum())
                      for ba, bb in zip(masks_before_alignment, c.masks) for a, b in zip(ba, bb))
        method_f = float(c.calculate_fitness([c.best])[0])
        emit("final_consistency",
             kind="resnet", candidate_id=inf["candidate_id"], rates=inf["rates"],
             implementation_legal=inf["implementation_legal"], diagnostic_bounds_ok=inf["diagnostic_bounds_ok"],
             candidate_pruned_channels=inf["counts"],
             mask_pruned_channels_before_alignment=d0,
             mask_pruned_channels_after_alignment=d1,
             changed_mask_elements_by_alignment=changed,
             final_channel_pruning_fraction=safe_div(sum(d1), c.original_channels),
             fitness_score=method_f,
             fitness_recomputed_from_mask_before_alignment=f0,
             fitness_recomputed_from_final_mask=f1,
             fitness_relative_diff=safe_div(abs(method_f - f1), abs(method_f)),
             fitness_mask_consistent=bool(np.isclose(method_f, f1, rtol=1e-5, atol=0.0)),
             counts_match_before_alignment=(list(d0) == list(inf["counts"])),
             counts_match_after_alignment=(list(d1) == list(inf["counts"])),
             final_mask_hash=mask_id([[np.asarray(x).astype(np.uint8) for x in b] for b in c.masks]),
             note="アライメント（共有チャネルの票決）でマスクが変わると、Fitness が評価した集合と実際に消す集合が一致しない")


def model_built(original, pruned, kind, controller=None, error=None):
    if not STATE.enabled:
        return
    with timed():
        info = {"kind": kind, "model_build_success": error is None,
                "model_build_error": None if error is None else repr(error)}
        if error is None:
            p0 = sum(p.numel() for p in original.parameters())
            p1 = sum(p.numel() for p in pruned.parameters())
            info.update({"original_parameters": p0, "pruned_parameters": p1,
                         "actual_parameter_reduction": safe_div(p0 - p1, p0)})
            if kind == "vit" and hasattr(pruned, "get_complexity"):
                try:
                    f0 = controller.original_flops if controller is not None else original.get_complexity()
                    f1 = pruned.get_complexity(dim_rate=0)
                    info.update({"original_flops_get_complexity": f0, "pruned_flops_get_complexity": f1,
                                 "actual_flops_reduction": safe_div(f0 - f1, f0),
                                 "target": getattr(controller, "percentage", None)})
                except Exception as e:  # 観測のみ。失敗理由を残す
                    info["flops_error"] = repr(e)
            info["actual_model_dimensions"] = {
                n: list(p.shape) for n, p in pruned.named_parameters()
                if n.endswith("weight") and ("qkv" in n or "fc1" in n or "patch_embed" in n or "conv" in n)}
        emit("model_build", **info)
