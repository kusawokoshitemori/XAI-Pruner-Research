"""
LRP の各規則の backward 内で呼ぶ観測ログ。

すべて lrp_active() のときだけ呼ばれる前提（呼び出し側で判定する）。
受け取った tensor は読むだけで、手法が返す関連度は一切変更しない。
"""
import torch

from .core import (STATE, emit, tensor_summary, quantile_dict, per_sample_quantiles,
                   save_snapshot, safe_div, spearman, timed)


def _name(tag):
    if tag is None:
        return "unannotated"
    return tag.get("module", "unannotated")


def _nonfinite(t):
    return int((~torch.isfinite(t)).sum().item())


def _flat_per_sample(t):
    return t.detach().reshape(t.shape[0], -1).to(torch.float64)


def _top_share(abs_values, frac=0.01):
    """絶対量の上位 frac の要素が全体の何割を持つか（少数要素が分布を支配していないか）"""
    v = abs_values.reshape(-1)
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return None
    total = v.sum().item()
    if total == 0:
        return None
    k = max(1, int(v.numel() * frac))
    return torch.topk(v, k).values.sum().item() / total


# ----------------------------------------------------------------------------
# G1 / G2 : 残差加算点
# ----------------------------------------------------------------------------
def residual_add(tag, input_a, input_b, outputs, out_relevance, denom_used,
                 rel_a_raw, rel_b_raw, rel_a_used, rel_b_used):
    with timed():
        tag = tag or {}
        if tag.get("shortcut", "a") == "a":
            short, func = input_a, input_b
            r_s_raw, r_f_raw, r_s, r_f = rel_a_raw, rel_b_raw, rel_a_used, rel_b_used
        else:
            short, func = input_b, input_a
            r_s_raw, r_f_raw, r_s, r_f = rel_b_raw, rel_a_raw, rel_b_used, rel_a_used

        a = _flat_per_sample(short)
        b = _flat_per_sample(func)
        z = a + b
        a_n = torch.linalg.vector_norm(a, dim=1)
        b_n = torch.linalg.vector_norm(b, dim=1)
        z_n = torch.linalg.vector_norm(z, dim=1)
        dot = (a * b).sum(dim=1)

        ratio = torch.where(b_n > 0, a_n / b_n.clamp_min(1e-300), torch.full_like(a_n, float("nan")))
        cos_ok = (a_n > 0) & (b_n > 0)
        cosine = torch.where(cos_ok, dot / (a_n * b_n).clamp_min(1e-300), torch.full_like(a_n, float("nan")))

        scale = a.abs() + b.abs()
        valid = scale > 0
        cancel = (z.abs()[valid] / scale[valid])
        opposite = (((a > 0) & (b < 0)) | ((a < 0) & (b > 0))).sum(dim=1).to(torch.float64)

        # 分母（加算出力）
        raw_d = outputs.detach().to(torch.float64)
        used_d = denom_used.detach().to(torch.float64)
        thr = STATE.near_zero_threshold
        near = raw_d.abs() < thr

        # 分岐ごとの絶対関連度（per sample）。符号付き比は 0〜1 の割合にならないので絶対量で内訳を出す
        rs = _flat_per_sample(r_s).abs().sum(dim=1)
        rf = _flat_per_sample(r_f).abs().sum(dim=1)
        q_func_abs = torch.where((rs + rf) > 0, rf / (rs + rf).clamp_min(1e-300),
                                 torch.full_like(rs, float("nan")))

        abs_rf = r_f.detach().abs().to(torch.float64)
        total_rf = abs_rf[torch.isfinite(abs_rf)].sum().item()
        rf_near = abs_rf[near & torch.isfinite(abs_rf)].sum().item()

        ratio_x_over_d = (short.detach().to(torch.float64) / used_d).abs()
        ratio_y_over_d = (func.detach().to(torch.float64) / used_d).abs()

        nonfinite_before = _nonfinite(r_s_raw) + _nonfinite(r_f_raw)
        snap = None
        if nonfinite_before:
            snap = save_snapshot("residual_nonfinite_" + _name(tag), input_a=input_a, input_b=input_b,
                                 out_relevance=out_relevance, denom_used=denom_used)

        emit("residual_add",
             module_id=_name(tag), branch_id=tag.get("branch"), shortcut_input=tag.get("shortcut"),
             activation={
                 "shortcut_l2": per_sample_quantiles(a_n),
                 "functional_l2": per_sample_quantiles(b_n),
                 "sum_l2": per_sample_quantiles(z_n),
                 "shortcut_rms": per_sample_quantiles(a.square().mean(dim=1).sqrt()),
                 "functional_rms": per_sample_quantiles(b.square().mean(dim=1).sqrt()),
                 "norm_ratio_shortcut_over_functional": per_sample_quantiles(ratio),
                 "cosine": per_sample_quantiles(cosine),
                 "inner_product_negative_sample_count": int((dot < 0).sum().item()),
                 "energy_change": per_sample_quantiles(z_n.square() - a_n.square()),
                 "norm_decreased_sample_count": int((z_n < a_n).sum().item()),
                 "opposite_sign_fraction": per_sample_quantiles(opposite / a.shape[1]),
                 "both_zero_count": int(((a == 0) & (b == 0)).sum().item()),
                 "cancellation_ratio_|x+y|/(|x|+|y|)": quantile_dict(cancel) if cancel.numel() else None,
                 "sample_count": a.shape[0],
                 "element_count_per_sample": a.shape[1],
             },
             incoming_relevance=tensor_summary(out_relevance),
             denominator={
                 "raw": tensor_summary(raw_d),
                 "used": tensor_summary(used_d),
                 "zero_count": int((raw_d == 0).sum().item()),
                 "near_zero_count": int(near.sum().item()),
                 "near_zero_threshold": thr,
             },
             shortcut_relevance_before_repair=tensor_summary(r_s_raw),
             functional_relevance_before_repair=tensor_summary(r_f_raw),
             shortcut_relevance_after_repair=tensor_summary(r_s),
             functional_relevance_after_repair=tensor_summary(r_f),
             abs_x_over_d=quantile_dict(ratio_x_over_d[torch.isfinite(ratio_x_over_d)]),
             abs_y_over_d=quantile_dict(ratio_y_over_d[torch.isfinite(ratio_y_over_d)]),
             functional_abs_share_q_y_abs=per_sample_quantiles(q_func_abs),
             note_q_y_abs="二分岐に生成された絶対量の内訳。出力関連度の保存割合ではない",
             # 「ノルム比が大きいと functional 関連度が小さい」のか「近ゼロ分母が支配する」のかを分ける
             spearman_norm_ratio_vs_q_y_abs_across_samples=spearman(ratio.cpu().numpy(), q_func_abs.cpu().numpy()),
             functional_abs_share_from_near_zero_denominator=safe_div(rf_near, total_rf),
             functional_abs_share_top1pct_elements=_top_share(abs_rf),
             nonfinite_before_repair=nonfinite_before,
             snapshot=snap)


# ----------------------------------------------------------------------------
# F1 / F3 / F4 / G3 : Filter（top_k=0 は Gate、0<k<1 は Filtering）
# ----------------------------------------------------------------------------
def gate(tag, top_k, out_relevance, returned):
    with timed():
        emit("gate",
             module_id=_name(tag), role=(tag or {}).get("role"), keep_fraction=top_k,
             received=tensor_summary(out_relevance),
             returned=tensor_summary(returned))


def passthrough_filter(tag, top_k, out_relevance):
    with timed():
        emit("filter_passthrough", module_id=_name(tag), role=(tag or {}).get("role"),
             keep_fraction=top_k, received=tensor_summary(out_relevance, quantiles=False))


def _near_zero_vs_filter(kept, z):
    """Filter の保持マスクと、対応する次の逆伝播演算の分母 z（= Filter の forward 入力）の対応"""
    if z is None:
        return {"status": "unmapped", "reason": "forward input not saved"}
    if z.shape != kept.shape:
        return {"status": "unmapped", "reason": "shape mismatch {} vs {}".format(list(z.shape), list(kept.shape))}
    thr = STATE.near_zero_threshold
    near = z.detach().abs() < thr
    removed = ~kept
    nz_total = int(near.sum().item())
    nz_removed = int((near & removed).sum().item())
    removed_total = int(removed.sum().item())
    return {
        "status": "mapped",
        "near_zero_threshold": thr,
        "near_zero_total": nz_total,
        "removed_total": removed_total,
        "near_zero_and_removed": nz_removed,
        "near_zero_but_kept": nz_total - nz_removed,
        "not_near_zero_but_removed": removed_total - nz_removed,
        "removal_rate_of_near_zero": safe_div(nz_removed, nz_total),
        "near_zero_content_of_removed": safe_div(nz_removed, removed_total),
        "near_zero_rate_overall": safe_div(nz_total, kept.numel()),
        "zero_denominator_count": int((z == 0).sum().item()),
        "note": "近ゼロは診断ラベルであり誤計算の確定ラベルではない",
    }


def filter_transformer(tag, top_k, out_relevance, topk_indices, masked, original_sum,
                       selected_sum, rescaled, forward_input):
    with timed():
        B = out_relevance.shape[0]
        kept = torch.zeros(B, out_relevance[0].numel(), dtype=torch.bool, device=out_relevance.device)
        kept.scatter_(1, topk_indices, True)
        kept = kept.view_as(out_relevance)

        scale = original_sum / selected_sum
        abs_before = out_relevance.detach().abs().to(torch.float64)
        removed_abs = abs_before[~kept].sum().item()

        nonfinite_after = _nonfinite(rescaled)
        snap = None
        if nonfinite_after:
            snap = save_snapshot("filter_nonfinite_" + _name(tag), out_relevance=out_relevance,
                                 masked=masked, original_sum=original_sum, selected_sum=selected_sum)

        emit("filter_rescale",
             module_id=_name(tag), role=(tag or {}).get("role"), keep_fraction=top_k,
             selection_scope="per_sample_flattened(N*C)",
             normalization_scope="per_sample_signed_sum",
             before=tensor_summary(out_relevance),
             after_mask=tensor_summary(masked),
             after_rescale=tensor_summary(rescaled),
             scale=tensor_summary(scale),
             scale_numerator_original_signed_sum=tensor_summary(original_sum, quantiles=False),
             scale_denominator_selected_signed_sum=tensor_summary(selected_sum),
             zero_denominator_count=int((selected_sum == 0).sum().item()),
             negative_scale_count=int((scale < 0).sum().item()),
             abs_scale_gt_10_count=int((scale.abs() > 10).sum().item()),
             selected_mask_count=int(kept.sum().item()),
             nonzero_after_mask_count=int((masked != 0).sum().item()),
             total_count=kept.numel(),
             removed_abs_fraction=safe_div(removed_abs, abs_before.sum().item()),
             nonfinite_after_rescale=nonfinite_after,
             near_zero_denominator=_near_zero_vs_filter(kept, forward_input),
             denominator_mapping=(tag or {}).get("denominator_mapping"),
             snapshot=snap)


def filter_cnn(tag, top_k, out_relevance, kept_flat, masked, original_sum, selected_sum,
               scale, rescaled_raw, rescaled_used, forward_input):
    with timed():
        kept = kept_flat.view_as(out_relevance)
        abs_before = out_relevance.detach().abs().to(torch.float64)
        removed_abs = abs_before[~kept].sum().item()

        # チャネル（サンプル×チャネル）単位で全空間要素が除去されたか
        kept_per_channel = kept.flatten(2).sum(dim=2)                  # [B, C]
        all_removed = kept_per_channel == 0
        orig_ch = original_sum.reshape(kept_per_channel.shape)
        nonfinite_before_repair = _nonfinite(rescaled_raw)
        snap = None
        if nonfinite_before_repair:
            snap = save_snapshot("filter_cnn_nonfinite_" + _name(tag), out_relevance=out_relevance,
                                 original_sum=original_sum, selected_sum=selected_sum)

        emit("filter_rescale",
             module_id=_name(tag), role=(tag or {}).get("role"), keep_fraction=top_k,
             selection_scope="whole_batch_flattened(B*C*H*W)",
             normalization_scope="per_sample_per_channel_spatial_abs_sum",
             before=tensor_summary(out_relevance),
             after_mask=tensor_summary(masked),
             after_rescale_before_repair=tensor_summary(rescaled_raw),
             after_rescale=tensor_summary(rescaled_used),
             scale=tensor_summary(scale),
             scale_denominator_selected_abs_sum=tensor_summary(selected_sum),
             zero_denominator_count=int((selected_sum == 0).sum().item()),
             channels_total=int(all_removed.numel()),
             channels_all_spatial_removed=int(all_removed.sum().item()),
             channels_all_removed_with_nonzero_original=int((all_removed & (orig_ch != 0)).sum().item()),
             kept_per_sample_fraction=per_sample_quantiles(
                 kept.flatten(1).sum(dim=1).to(torch.float64) / kept[0].numel()),
             note_batch_dependence="Top-k はバッチ全体で選ぶため、同じサンプルでも同居サンプルで保持マスクが変わり得る",
             selected_mask_count=int(kept.sum().item()),
             nonzero_after_mask_count=int((masked != 0).sum().item()),
             total_count=kept.numel(),
             removed_abs_fraction=safe_div(removed_abs, abs_before.sum().item()),
             nonfinite_before_repair=nonfinite_before_repair,
             near_zero_denominator=_near_zero_vs_filter(kept, forward_input),
             denominator_mapping=(tag or {}).get("denominator_mapping"),
             snapshot=snap)


# ----------------------------------------------------------------------------
# B1 / B2 / 7.6 : epsilon Linear（バイアス再分配・符号反転・保存）
# ----------------------------------------------------------------------------
def _sign_change(data, total, thr):
    valid = torch.isfinite(data) & torch.isfinite(total) & (data != 0) & (total != 0)
    flipped = valid & (torch.sign(data) != torch.sign(total))
    big = valid & (data.abs() >= thr) & (total.abs() >= thr)
    flipped_big = big & (torch.sign(data) != torch.sign(total))
    return {
        "comparable_count": int(valid.sum().item()),
        "flipped_count": int(flipped.sum().item()),
        "flip_rate": safe_div(flipped.sum().item(), valid.sum().item()),
        "comparable_count_above_threshold": int(big.sum().item()),
        "flipped_count_above_threshold": int(flipped_big.sum().item()),
        "threshold": thr,
        "zero_to_nonzero": int(((data == 0) & (total != 0)).sum().item()),
        "nonzero_to_zero": int(((data != 0) & (total == 0)).sum().item()),
    }


def linear(tag, inputs, outputs_raw, denom_stabilized, denom_used, out_relevance,
           data_raw, data_used, bias_term, combined_raw, combined_used):
    with timed():
        thr = STATE.near_zero_threshold
        raw = outputs_raw.detach()
        returned = combined_used if combined_used is not None else data_used

        info = {
            "module_id": _name(tag),
            "has_bias": bias_term is not None,
            "denominator": {
                "raw": tensor_summary(raw),
                "used": tensor_summary(denom_used),
                "raw_zero_count": int((raw == 0).sum().item()),
                "raw_near_zero_count": int((raw.abs() < thr).sum().item()),
                "changed_by_clamp_count": int((denom_used != denom_stabilized).sum().item()),
                "near_zero_threshold": thr,
            },
            "incoming_relevance": tensor_summary(out_relevance),
            "data_raw": tensor_summary(data_raw),
            "data_used": tensor_summary(data_used),
            "data_nonfinite_before_repair": _nonfinite(data_raw),
            "returned": tensor_summary(returned),
        }

        # F2: ゼロ活性 → 非ゼロ帰属（バイアス再分配で 0 入力にも関連度が付き得る）
        x = inputs.detach()
        x0 = x == 0
        info["input_activation"] = {
            "zero_count": int(x0.sum().item()),
            "zero_with_nonzero_returned_relevance": int((x0 & (returned != 0)).sum().item()),
            "zero_with_nonzero_data_term": int((x0 & (data_used != 0)).sum().item()),
            "numel": x.numel(),
        }

        if bias_term is not None:
            D = data_used.detach().to(torch.float64)
            Bt = bias_term.detach().to(torch.float64).expand_as(D)
            T = combined_used.detach().to(torch.float64)
            sumD = D.abs().sum().item()
            sumB = Bt.abs().sum().item()
            info.update({
                "bias_term_per_position": tensor_summary(bias_term),
                "combined_raw": tensor_summary(combined_raw),
                "combined_nonfinite_before_repair": _nonfinite(combined_raw),
                "combined_used": tensor_summary(combined_used),
                "sign_change_data_vs_total": _sign_change(D, T, STATE.flip_threshold),
                "abs_change_||T|-|D||": quantile_dict((T.abs() - D.abs()).abs()[torch.isfinite(T) & torch.isfinite(D)]),
                "abs_change_relative_to_sum_abs_D": safe_div(
                    (T.abs() - D.abs()).abs()[torch.isfinite(T) & torch.isfinite(D)].sum().item(), sumD),
                "cancellation_C=1-sum|D+B|/(sum|D|+sum|B|)": (
                    None if (sumD + sumB) == 0 else 1.0 - (D + Bt).abs().sum().item() / (sumD + sumB)),
                "bias_abs_share=sum|B|/(sum|D|+sum|B|)": safe_div(sumB, sumD + sumB),
            })

        # 保存則（局所）。nan_to_num や clamp の影響を含むので、どれが効いたかは他の項目と併せて読む
        s_in = out_relevance.detach().to(torch.float64)
        s_out = returned.detach().to(torch.float64)
        sin = s_in[torch.isfinite(s_in)].sum().item()
        sout = s_out[torch.isfinite(s_out)].sum().item()
        info["conservation"] = {
            "incoming_signed_sum": sin,
            "returned_signed_sum": sout,
            "data_only_signed_sum": data_used.detach().to(torch.float64).sum().item(),
            "relative_diff": safe_div(sout - sin, abs(sin)),
        }

        if info["data_nonfinite_before_repair"] or info.get("combined_nonfinite_before_repair"):
            info["snapshot"] = save_snapshot("linear_nonfinite_" + _name(tag), inputs=inputs,
                                             outputs_raw=outputs_raw, out_relevance=out_relevance)
        emit("linear_lrp", **info)


def conv2d(tag, inputs, outputs_raw, denom_used, out_relevance, rel_raw, rel_used):
    with timed():
        thr = STATE.near_zero_threshold
        raw = outputs_raw.detach()
        sin = out_relevance.detach().to(torch.float64)
        sout = rel_used.detach().to(torch.float64)
        nf = _nonfinite(rel_raw)
        snap = None
        if nf:
            snap = save_snapshot("conv_nonfinite_" + _name(tag), inputs=inputs, outputs_raw=outputs_raw,
                                 out_relevance=out_relevance)
        x0 = inputs.detach() == 0
        emit("conv2d_lrp",
             module_id=_name(tag),
             denominator={
                 "raw": tensor_summary(raw),
                 "used": tensor_summary(denom_used),
                 "raw_zero_count": int((raw == 0).sum().item()),
                 "raw_near_zero_count": int((raw.abs() < thr).sum().item()),
                 "near_zero_threshold": thr,
             },
             incoming_relevance=tensor_summary(out_relevance),
             relevance_before_repair=tensor_summary(rel_raw),
             returned=tensor_summary(rel_used),
             nonfinite_before_repair=nf,
             input_activation={"zero_count": int(x0.sum().item()),
                               "zero_with_nonzero_returned_relevance": int((x0 & (rel_used != 0)).sum().item()),
                               "numel": x0.numel()},
             conservation={
                 "incoming_signed_sum": sin[torch.isfinite(sin)].sum().item(),
                 "returned_signed_sum": sout[torch.isfinite(sout)].sum().item(),
                 "relative_diff": safe_div(sout[torch.isfinite(sout)].sum().item() - sin[torch.isfinite(sin)].sum().item(),
                                           abs(sin[torch.isfinite(sin)].sum().item())),
             },
             snapshot=snap)


# ----------------------------------------------------------------------------
# G4 : BN の恒等伝播の前提（モジュールごとに一度だけ）
# ----------------------------------------------------------------------------
_bn_logged = set()


def batchnorm_config(name, bn, x):
    if name in _bn_logged:
        return
    _bn_logged.add(name)
    with timed():
        uses_batch_stats = bn.training or not bn.track_running_stats or bn.running_mean is None
        info = {
            "module_id": name,
            "training": bn.training,
            "track_running_stats": bn.track_running_stats,
            "has_running_mean": bn.running_mean is not None,
            "has_running_var": bn.running_var is not None,
            "uses_batch_statistics_in_forward": bool(uses_batch_stats),
            "input_shape": list(x.shape),
            "relevance_rule": "identity (incoming relevance returned unchanged, incl. shift term)",
        }
        if bn.running_var is not None and bn.weight is not None:
            a = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
            b = (bn.bias - bn.running_mean * a).detach()
            info["scale_a"] = tensor_summary(a)
            info["shift_b"] = tensor_summary(b)
        emit("batchnorm_config", **info)


# ----------------------------------------------------------------------------
# モジュールへの名前付け（ログの module_id 用）
# ----------------------------------------------------------------------------
_MAPPING = {
    "filter_attn_in": ("filtering", "attn.proj の出力（間の proj_drop / drop_path1 は eval で恒等）"),
    "filter_mlp_in": ("filtering", "mlp.fc2 の出力（間の drop2 / drop_path2 は eval で恒等）"),
    "filter_attn_out": ("gate", None),
    "filter_mlp_out": ("gate", None),
    "residual_filter": ("gate", None),
}


def annotate(model):
    """各 LRP モジュールに diag_tag を付ける。forward の計算には影響しない"""
    from lrp.module import Filter, EpsilonLinearLRP, Conv2dLRP, BatchNorm2dLRP

    for name, m in model.named_modules():
        if isinstance(m, Filter):
            leaf = name.split(".")[-1]
            role, mapping = _MAPPING.get(leaf, ("filtering", None))
            if leaf == "filter":  # Conv2dLRP.filter
                role, mapping = "filtering", "親 Conv2d の出力（直接）"
            m.diag_tag = {"module": name, "role": role, "denominator_mapping": mapping}
        elif isinstance(m, (EpsilonLinearLRP, Conv2dLRP)):
            m.diag_tag = {"module": name}
        elif isinstance(m, BatchNorm2dLRP):
            m.diag_tag = {"module": name}

        # 残差加算の呼出側（ViT Block / ResNet Block）
        cls = type(m).__name__
        if cls == "BlockLRP":
            m.diag_tag_attn = {"module": name + ".attn_residual", "branch": "attn", "shortcut": "a"}
            m.diag_tag_mlp = {"module": name + ".mlp_residual", "branch": "mlp", "shortcut": "a"}
        elif cls in ("BasicBlockLRP", "BottleneckLRP"):
            # add_tensors_lrp.apply(x, residual): a = functional, b = shortcut
            m.diag_tag_add = {"module": name + ".residual", "branch": "block", "shortcut": "b"}

    # 0 段階: Filter の実行時設定（--filter_percent が実際に反映されているか）
    filters = [{"module": n, "keep_fraction": f.top_k, "role": getattr(f, "diag_tag", {}).get("role")}
               for n, f in model.named_modules() if isinstance(f, Filter)]
    by_value = {}
    for f in filters:
        by_value.setdefault("{}:{}".format(f["role"], f["keep_fraction"]), []).append(f["module"])
    emit("filter_runtime_config",
         count=len(filters),
         grouped={k: {"count": len(v), "examples": v[:3]} for k, v in by_value.items()},
         filters=filters)
