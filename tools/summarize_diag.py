"""
診断ログ（--diag_dir の出力）を、疑問 ID（G1〜I1）ごとの表に集計して summary.md を作る。

使い方:
    python tools/summarize_diag.py <diag_dir>/<run_id>/rank0

ここで出す値はすべて記録された実測値からの集計（status = measured）。
理論値と並べる箇所は「theory」と明記する。
"""
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np


def load_events(root):
    path = os.path.join(root, "events.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_csv(root, name):
    path = os.path.join(root, name + ".csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def b(v):
    return str(v).lower() == "true"


def num(v):
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return float("nan")


def ratio(a, c):
    return "NA" if not c else "{:.3f} ({}/{})".format(a / c, a, c)


def fmt(x, d=4):
    if x is None:
        return "NA"
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return "{:.3e}".format(x)
        return "{:.{}f}".format(x, d)
    return str(x)


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else None


def get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


class Md(object):
    def __init__(self):
        self.lines = []

    def h(self, t, level=2):
        self.lines += ["", "#" * level + " " + t, ""]

    def p(self, t):
        self.lines.append(t)

    def table(self, header, rows):
        if not rows:
            self.p("（記録なし）")
            return
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "---|" * len(header))
        for r in rows:
            self.lines.append("| " + " | ".join(fmt(x) for x in r) + " |")

    def text(self):
        return "\n".join(self.lines) + "\n"


def by_module(events, name):
    out = defaultdict(list)
    for e in events:
        if e["event"] == name:
            out[e.get("module_id")].append(e)
    return out


def main(root):
    ev = load_events(root)
    md = Md()
    meta_path = os.path.join(root, "metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    md.p("# 診断ログ集計: {}".format(meta.get("run_id", root)))
    md.p("")
    md.p("全項目 status = measured（記録値からの集計）。理論値は theory と明記。")

    # ------------------------------------------------------------------ 0段階
    md.h("第0段階: 実行環境・設定の整合性")
    seeds = meta.get("seeds", {})
    md.table(["項目", "値"], [
        ["python", (meta.get("python") or "").split()[0] if meta.get("python") else None],
        ["torch / numpy", "{} / {}".format(meta.get("torch"), meta.get("numpy"))],
        ["git commit (dirty)", "{} ({})".format(get(meta, "git", "commit"), get(meta, "git", "dirty"))],
        ["Python random の seed をスクリプトが設定", seeds.get("python_random_seeded_by_script")],
        ["random.sample(set) が動く版か (<3.11)", meta.get("random_sample_accepts_set")],
    ])
    for e in ev:
        if e["event"] == "filter_runtime_config":
            md.p("")
            md.p("Filter の実行時設定（role:keep_fraction → 個数）")
            md.table(["role:keep", "count", "例"], [[k, v["count"], ", ".join(v["examples"])] for k, v in e["grouped"].items()])
        if e["event"] == "protect_check":
            md.p("")
            md.p("protect: 層あたり要求保護数 {} / Inf になった Head {} / 全 Head {}  {}".format(
                e["requested_protect_count_per_layer"], e["head_inf_after"], e["head_total"], e.get("note") or ""))
        if e["event"] == "lrp_model_mode":
            md.p("")
            md.p("関連度計算時に training=True のモジュール数: {}".format(e["training_module_count"]))

    # ------------------------------------------------------------------ G1/G2
    md.h("G1/G2: 残差加算点（活性ノルムと関連度配分）")
    md.p("各値はサンプル間中央値をバッチ間で中央値にしたもの。q_y_abs = Σ|R_func| / (Σ|R_short|+Σ|R_func|)（絶対量の内訳であり保存割合ではない）")
    rows = []
    for m, es in by_module(ev, "residual_add").items():
        rows.append([m,
                     med([get(e, "activation", "shortcut_rms", "q50") for e in es]),
                     med([get(e, "activation", "functional_rms", "q50") for e in es]),
                     med([get(e, "activation", "norm_ratio_shortcut_over_functional", "q50") for e in es]),
                     med([get(e, "activation", "cosine", "q50") for e in es]),
                     sum(get(e, "activation", "inner_product_negative_sample_count") or 0 for e in es),
                     med([get(e, "functional_abs_share_q_y_abs", "q50") for e in es]),
                     med([e.get("spearman_norm_ratio_vs_q_y_abs_across_samples") for e in es]),
                     med([e.get("functional_abs_share_from_near_zero_denominator") for e in es]),
                     med([e.get("functional_abs_share_top1pct_elements") for e in es]),
                     sum(get(e, "denominator", "near_zero_count") or 0 for e in es),
                     sum(e.get("nonfinite_before_repair") or 0 for e in es)])
    md.table(["module", "shortcut_rms", "functional_rms", "norm比 s/f", "cos", "内積<0 sample数",
              "q_y_abs", "ρ(norm比,q_y_abs)", "func|R|のうち近ゼロ分母由来", "func|R|上位1%要素の占有",
              "近ゼロ分母数", "修復前非有限"], rows)

    # ------------------------------------------------------------------ G3
    md.h("G3: Gate（分岐入口で受け取った関連度と返した関連度）")
    rows = []
    for m, es in by_module(ev, "gate").items():
        rows.append([m, med([get(e, "received", "finite_abs_sum") for e in es]),
                     med([get(e, "received", "finite_signed_sum") for e in es]),
                     max(get(e, "returned", "finite_abs_sum") or 0 for e in es)])
    md.table(["module", "受信 abs_sum", "受信 signed_sum", "返却 abs_sum(max)"], rows)

    # ------------------------------------------------------------------ G4
    bns = [e for e in ev if e["event"] == "batchnorm_config"]
    if bns:
        md.h("G4: BatchNorm の恒等伝播の前提")
        md.p("BN 数 {} / forward でバッチ統計量を使うもの {} / training=True {}".format(
            len(bns), sum(e["uses_batch_statistics_in_forward"] for e in bns), sum(e["training"] for e in bns)))

    # ------------------------------------------------------------------ F1/F3/F4
    md.h("F1/F3/F4: Filtering（マスク・再スケーリング・近ゼロ分母との対応）")
    rows, rows_nz = [], []
    for m, es in by_module(ev, "filter_rescale").items():
        rows.append([m, es[0].get("keep_fraction"), es[0].get("selection_scope"),
                     med([e.get("removed_abs_fraction") for e in es]),
                     med([get(e, "scale", "finite_abs_quantiles", "q50") for e in es]),
                     max((get(e, "scale", "finite_max_abs") or 0) for e in es),
                     sum(e.get("negative_scale_count") or 0 for e in es),
                     sum(e.get("zero_denominator_count") or 0 for e in es),
                     sum((e.get("nonfinite_after_rescale") or e.get("nonfinite_before_repair") or 0) for e in es),
                     ratio(sum(e.get("nonzero_after_mask_count") or 0 for e in es),
                           sum(e.get("selected_mask_count") or 0 for e in es)),
                     sum(e.get("channels_all_spatial_removed") or 0 for e in es) if "channels_total" in es[0] else None])
        nz = [e.get("near_zero_denominator") or {} for e in es]
        if nz and nz[0].get("status") == "mapped":
            t = lambda k: sum(x.get(k) or 0 for x in nz)
            rows_nz.append([m, t("near_zero_total"), t("removed_total"), t("near_zero_and_removed"),
                            t("near_zero_but_kept"), t("not_near_zero_but_removed"),
                            ratio(t("near_zero_and_removed"), t("near_zero_total")),
                            ratio(t("near_zero_and_removed"), t("removed_total"))])
    md.table(["module", "keep", "選別範囲", "除去した|R|割合", "scale中央値|.|", "scale最大|.|",
              "負scale数", "scale分母0数", "非有限(修復前)", "非ゼロ/選択数", "全空間除去ch数"], rows)
    md.p("")
    md.p("F1: 近ゼロ分母（Filter の forward 入力 = 次に関連度を戻す演算の分母）と除去の対応")
    md.table(["module", "近ゼロ総数", "除去総数", "近ゼロ∧除去", "近ゼロだが保持", "近ゼロでないが除去",
              "近ゼロ経路の除去率", "除去集合の近ゼロ含有率"], rows_nz)

    # ------------------------------------------------------------------ B1/B2/7.6
    md.h("B1/B2/7.6: epsilon Linear（バイアス再分配・符号反転・保存）")
    rows = []
    for m, es in by_module(ev, "linear_lrp").items():
        sc = [e.get("sign_change_data_vs_total") or {} for e in es]
        rows.append([m,
                     ratio(sum(x.get("flipped_count", 0) for x in sc), sum(x.get("comparable_count", 0) for x in sc)) if sc[0] else None,
                     ratio(sum(x.get("flipped_count_above_threshold", 0) for x in sc),
                           sum(x.get("comparable_count_above_threshold", 0) for x in sc)) if sc[0] else None,
                     sum(x.get("zero_to_nonzero", 0) for x in sc),
                     med([e.get("cancellation_C=1-sum|D+B|/(sum|D|+sum|B|)") for e in es]),
                     med([e.get("bias_abs_share=sum|B|/(sum|D|+sum|B|)") for e in es]),
                     med([get(e, "conservation", "relative_diff") for e in es]),
                     sum(get(e, "denominator", "raw_near_zero_count") or 0 for e in es),
                     sum(get(e, "denominator", "changed_by_clamp_count") or 0 for e in es),
                     sum((e.get("data_nonfinite_before_repair") or 0) + (e.get("combined_nonfinite_before_repair") or 0) for e in es),
                     sum(get(e, "input_activation", "zero_with_nonzero_returned_relevance") or 0 for e in es)])
    md.table(["module", "符号反転率", "符号反転率(閾値以上)", "0→非0", "相殺C", "バイアス|.|占有",
              "保存相対差", "近ゼロ分母数", "clamp変更数", "修復前非有限", "0入力に非0関連度"], rows)

    rows = []
    for m, es in by_module(ev, "conv2d_lrp").items():
        rows.append([m, sum(get(e, "denominator", "raw_near_zero_count") or 0 for e in es),
                     sum(e.get("nonfinite_before_repair") or 0 for e in es),
                     med([get(e, "conservation", "relative_diff") for e in es]),
                     sum(get(e, "input_activation", "zero_with_nonzero_returned_relevance") or 0 for e in es)])
    if rows:
        md.p("")
        md.p("Conv2d")
        md.table(["module", "近ゼロ分母数", "修復前非有限", "保存相対差", "0入力に非0関連度"], rows)

    # ------------------------------------------------------------------ S1/S2
    comp = load_csv(root, "component_scores")
    if comp:
        md.h("S1/S2: 構造種類別のスコアと集約サイズ（protect 前）")
        for e in ev:
            if e["event"] == "component_size_definition":
                md.p("n（集約要素数）: head={n_head}, hidden={n_hidden}, dim={n_dim} / g（単独削除パラメータ数）: head={g_head}, hidden={g_hidden}, dim={g_dim}".format(**e))
                md.p("g の定義: " + e["parameter_count_definition"])
        g = defaultdict(list)
        for r in comp:
            g[r["component_type"]].append(r)
        rows = []
        for t, rs in g.items():
            s = np.array([num(r["score_sum_abs"]) for r in rs])
            rows.append([t, len(rs), float(np.median(s)), float(np.quantile(s, 0.1)), float(np.quantile(s, 0.9)),
                         med([num(r["score_per_aggregated_element"]) for r in rs]),
                         med([num(r["score_per_parameter"]) for r in rs]),
                         med([num(r["mean_nonzero_element_count"]) for r in rs])])
        md.table(["種類", "構造数", "S中央値", "S q10", "S q90", "S/n 中央値", "S/g 中央値", "非ゼロ要素数中央値"], rows)

    cands = load_csv(root, "candidate_evaluations")
    if cands:
        last = max(int(r["epoch"]) for r in cands)
        sel = [r for r in cands if int(r["epoch"]) == last and b(r["selected"])]
        if sel:
            md.p("")
            md.p("8.5: 最終世代の選択候補での Fitness 内訳（中央値）")
            shares = defaultdict(list)
            for r in sel:
                for k, v in json.loads(r["fitness_part_shares"]).items():
                    shares[k].append(v)
            md.table(["項", "F_total に占める割合（中央値）"], [[k, med(v)] for k, v in shares.items()])
        gens = [e for e in ev if e["event"] == "ea_generation"]
        if gens:
            md.p("")
            md.p("候補間で、各項の順位と F_total 順位の Spearman（最終世代のプール）")
            md.table(["項", "ρ"], [[k, v] for k, v in (gens[-1].get("spearman_fitness_part_vs_total") or {}).items()])

    # ------------------------------------------------------------------ 11.8
    for e in ev:
        if e["event"] == "score_aggregation_comparison":
            md.h("11.8: momentum 集計 vs 単純平均")
            md.p("バッチ数 {} / momentum {} / 最初のバッチの最終重み {}".format(
                e["batches"], e["momentum"], fmt(e["final_weight_of_first_batch"])))
            md.table(["group", "Spearman", "下位10%一致", "下位30%一致", "下位50%一致"],
                     [[k, v.get("spearman_momentum_vs_plain_mean"), v.get("lowest_10pct_overlap"),
                       v.get("lowest_30pct_overlap"), v.get("lowest_50pct_overlap")] for k, v in e["by_group"].items()])

    init = load_csv(root, "initial_relevance")
    if init:
        md.h("11.9: 初期関連度（予測クラスの softmax 確率）")
        n = len(init)
        md.p("サンプル数 {} / 元モデル正解 {} / 関連度クラス=正解ラベル {} / 確信度中央値 {}".format(
            n, ratio(sum(b(r["baseline_correct"]) for r in init), n),
            ratio(sum(r["initial_relevance_class"] == r["true_label"] for r in init), n),
            fmt(med([num(r["prediction_confidence"]) for r in init]))))

    # ------------------------------------------------------------------ E1/E2/E3
    att = load_csv(root, "ea_attempts")
    calls = load_csv(root, "ea_operator_calls")
    if att:
        md.h("E1/E2: 演算子ごとの試行（全世代合計）")
        rows = []
        for op in ("crossover", "mutation"):
            a = [r for r in att if r["operator"] == op]
            if not a:
                continue
            n = len(a)
            legal = [r for r in a if b(r["legal_before_revise"]) or b(r["revised"])]
            added = [r for r in a if b(r["added_to_operator_kids"])]
            rows.append([op, n,
                         ratio(sum(b(r["legal_before_revise"]) for r in a), n),
                         ratio(sum(b(r["revised"]) for r in a), n),
                         ratio(sum(b(r["already_in_parent_pool"]) for r in a), n),
                         ratio(sum(b(r["already_in_parent_pool"]) for r in legal), len(legal)),
                         ratio(sum(b(r["already_in_parent_pool"]) for r in added), len(added)),
                         ratio(sum(b(r["already_in_operator_kids"]) for r in a), n),
                         ratio(sum(b(r["novel_genotype_added"]) for r in a), n),
                         ratio(sum(b(r["novel_mask_added"]) for r in a), n)])
        md.table(["operator", "試行", "制約通過率", "revise率", "親重複(全試行)", "親重複(合法試行)",
                  "親重複(kids追加)", "kids既存", "新規候補/試行", "新規マスク/試行"], rows)
        md.p("")
        md.p("E3: 停止理由（名目要求数に到達 ≠ 十分な新規モデル）")
        rows = []
        for op in ("crossover", "mutation"):
            c = [r for r in calls if r["operator"] == op]
            if not c:
                continue
            st = Counter(r["stop_reason"] for r in c)
            rows.append([op, len(c), st.get("requested_kids_reached", 0), st.get("max_iters_reached", 0),
                         med([num(r["attempts"]) for r in c]),
                         med([num(r["returned_unique_genotypes"]) for r in c]),
                         med([num(r["novel_unique_genotypes"]) for r in c]),
                         med([num(r["novel_unique_masks"]) for r in c]),
                         sum(int(r["returned_illegal_or_out_of_bounds"]) for r in c)])
        md.table(["operator", "呼出数", "要求数到達", "max_iters到達", "試行数中央値", "返却一意候補中央値",
                  "新規一意候補中央値", "新規一意マスク中央値", "返却中の非合法/範囲外"], rows)

        md.h("10.3: 交叉の親被り（実測 vs theory 2^(1-d)）")
        rows = []
        cx = [r for r in att if r["operator"] == "crossover"]
        for d in sorted({r["parent_gene_difference_count"] for r in cx}, key=lambda x: int(x)):
            a = [r for r in cx if r["parent_gene_difference_count"] == d]
            same = sum(b(r["child_same_as_parent1"]) or b(r["child_same_as_parent2"]) for r in a)
            same_m = sum(b(r["child_mask_same_as_parent1"]) or b(r["child_mask_same_as_parent2"]) for r in a)
            di = int(d)
            rows.append([di, len(a), ratio(same, len(a)), 2 ** (1 - di) if di >= 1 else 1.0, ratio(same_m, len(a))])
        md.table(["異なる遺伝子数 d", "試行", "親と同一(実測)", "theory", "親と同一マスク(実測)"], rows)

        md.h("10.4: 突然変異の変異数別（実測 vs theory Binomial）")
        mu = [r for r in att if r["operator"] == "mutation"]
        if mu:
            G = len(json.loads(mu[0]["mutation_decision_mask"]))
            rows = []
            for k in range(G + 1):
                a = [r for r in mu if int(r["mutated_gene_count"]) == k]
                theory = math.comb(G, k) * 0.25 ** k * 0.75 ** (G - k)
                rows.append([k, len(a), ratio(len(a), len(mu)), theory,
                             ratio(sum(b(r["legal_before_revise"]) for r in a), len(a)),
                             ratio(sum(b(r["same_as_source_parent"]) for r in a), len(a)),
                             ratio(sum(b(r["added_to_operator_kids"]) for r in a), len(a)),
                             ratio(sum(b(r["novel_genotype_added"]) for r in a), len(a)),
                             ratio(sum(b(r["novel_mask_added"]) for r in a), len(a))])
            md.p("theory は prob=0.25（engine の既定値）を仮定")
            md.table(["変異数", "試行", "割合", "theory", "制約通過", "親と同一", "kids追加", "新規候補", "新規マスク"], rows)

        rev = [r for r in att if b(r["revised"])]
        if rev:
            md.p("")
            md.p("11.3: revise() 後の合法性 {} / 範囲内 {}".format(
                ratio(sum(b(r["legal_after_revise"]) for r in rev), len(rev)),
                ratio(sum(b(r["final_bounds_ok"]) for r in rev), len(rev))))

    # ------------------------------------------------------------------ E4/E5/E6
    gens = [e for e in ev if e["event"] == "ea_generation"]
    if gens:
        md.h("E4/E5/E6: Top-k 選択と生成源（全世代合計）")
        md.table(["項目", "値"], [
            ["世代数", len(gens)],
            ["選択割合（個体）中央値", med([e["selected_genotype_fraction"] for e in gens])],
            ["選択割合（マスク）中央値", med([e["selected_mask_fraction"] for e in gens])],
            ["選択内の重複マスク数 中央値", med([e["selected_duplicate_mask_count"] for e in gens])],
            ["プール一意マスク / プール候補 中央値", med([e["pool_unique_mask_count"] / e["pool_genotype_count"] for e in gens])],
            ["Top-k 境界の tie があった世代", sum(1 for e in gens if (e.get("topk_boundary_tie") or {}).get("tie_count", 0) > 1)],
            ["Fitness 再計算との最大差", max(e["fitness_recompute_max_abs_diff"] for e in gens)],
            ["プール内 非合法/範囲外（合計）", sum(e["pool_illegal_or_out_of_bounds"] for e in gens)],
            ["選択内 非合法/範囲外（合計）", sum(e["selected_illegal_or_out_of_bounds"] for e in gens)],
        ])
        md.p("")
        rows = []
        for src in ("previous", "random", "cross_only", "mutation_only", "both_ops"):
            s = sum(e["sources"][src]["survived"] for e in gens)
            g_ = sum(e["sources"][src]["generated"] for e in gens)
            sel = sum(e["selected_genotype_count"] for e in gens)
            rows.append([src, g_, s, ratio(s, g_), ratio(s, sel)])
        md.p("候補ベクトル単位（生存率と Top-k 内構成比は分母が違う）")
        md.table(["生成源", "生成数", "生存数", "生存率", "Top-k内構成比"], rows)
        rows = []
        for src in ("existing", "random_only", "crossover_only", "mutation_only", "multiple_ops"):
            s = sum(e["mask_sources"][src]["survived"] for e in gens)
            g_ = sum(e["mask_sources"][src]["generated"] for e in gens)
            sel = sum(e["selected_unique_mask_count"] for e in gens)
            rows.append([src, g_, s, ratio(s, g_), ratio(s, sel)])
        md.p("")
        md.p("マスク単位")
        md.table(["生成源", "マスク数", "生存数", "生存率", "Top-k内構成比"], rows)

    rnd = load_csv(root, "random_generator")
    if rnd:
        md.h("11.4/11.5: ランダム補充")
        st = Counter(r["termination_reason"] for r in rnd)
        md.p("生成数 {} / 合法 {} / 範囲内 {} / 集合に追加された {}".format(
            len(rnd), ratio(sum(b(r["returned_implementation_legal"]) for r in rnd), len(rnd)),
            ratio(sum(b(r["returned_bounds_ok"]) for r in rnd), len(rnd)),
            ratio(sum(b(r["added_to_population"]) for r in rnd), len(rnd))))
        if any(k for k in st):
            md.table(["終了理由", "件数"], [[k, v] for k, v in st.items() if k])
        it = [num(r["binary_search_iterations"]) for r in rnd if r["binary_search_iterations"]]
        if it:
            md.p("二分探索反復数: 中央値 {} / 最大 {}".format(fmt(med(it)), fmt(max(it))))
        if any(e["event"] == "random_generator_long_loop" for e in ev):
            md.p("**random_generator_long_loop が記録されています（10万反復超）**")

    # ------------------------------------------------------------------ I1
    for e in ev:
        if e["event"] == "final_consistency":
            md.h("I1 / 11.7 / 12: 最良候補 → マスク → モデルの整合性")
            md.table(["項目", "値"], [[k, json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v]
                                     for k, v in e.items() if k not in ("run_id", "rank", "seq", "time_ns", "event", "status", "phase", "batch_id")])
        if e["event"] == "model_build":
            md.p("")
            md.table(["項目", "値"], [[k, v] for k, v in e.items()
                                     if k not in ("run_id", "rank", "seq", "time_ns", "event", "status", "phase", "batch_id", "actual_model_dimensions")])
        if e["event"] == "timing":
            md.h("15.4: 時間")
            md.p("全体 {:.1f}s / うちログ処理 {:.1f}s / 手法（推定） {:.1f}s".format(
                e["wall_seconds_since_init"], e["diag_seconds"], e["method_seconds_estimate"]))
            md.p(e["note"])

    out = os.path.join(root, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md.text())
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
