import torch
import torch.fx
from torch.autograd import Function
import torch.nn.functional as F
import math

import diag
from diag import relevance as diag_relevance


def _stabilize(input, epsilon=1e-6, inplace=False):
    """
    Stabilize the input by adding a small value to it
    """
    if inplace:
        return input.add_(epsilon * torch.sign(input))
    else:
        return input + epsilon * torch.sign(input)


class epsilon_linear_lrp(Function):
    """
    # Distribute the relevance value of the bias uniformly
    """
    @staticmethod
    def forward(ctx, inputs, weight, bias=None, epsilon=1e-6, tag=None):
        outputs = F.linear(inputs, weight, bias)
        ctx.save_for_backward(inputs, weight, bias, outputs)
        ctx.epsilon = epsilon
        ctx.diag_tag = tag

        return outputs

    @staticmethod
    def backward(ctx, *out_relevance):
        inputs, weight, bias, outputs = ctx.saved_tensors
        epsilon = ctx.epsilon
        raw_outputs = outputs

        outputs = _stabilize(outputs, epsilon)
        stabilized = outputs
        outputs = torch.where(outputs > 0, torch.clamp(outputs, min=ctx.epsilon), torch.clamp(outputs, max=-ctx.epsilon))

        relevance_norm = out_relevance[0] / outputs

        relevance = torch.matmul(relevance_norm, weight).mul_(inputs)
        data_raw = relevance

        relevance = torch.nan_to_num(relevance, nan=0.0, posinf=1e9, neginf=-1e9)
        data_used = relevance
        bias_relevance = combined_raw = None


        if bias is not None:
            N = weight.shape[1]
            bias_relevance = torch.sum((relevance_norm * bias / N), dim=-1, keepdim=True)

            bias_relevance = bias_relevance.expand_as(bias_relevance)
            relevance = relevance + bias_relevance
            combined_raw = relevance
            relevance = torch.nan_to_num(relevance, nan=0.0, posinf=1e9, neginf=-1e9)

        if diag.lrp_active():
            diag_relevance.linear(ctx.diag_tag, inputs, raw_outputs, stabilized, outputs, out_relevance[0],
                                  data_raw, data_used, bias_relevance, combined_raw,
                                  relevance if bias is not None else None)

        return relevance, None, None, None, None


class identity_lrp(Function):
    """
    Distribute the relevance 100% to the input according to the identity rule in Equation 9 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    Parameters:
    -----------

    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch and has a single input and output
    input: torch.Tensor
        The input tensor
    """

    @staticmethod
    def forward(ctx, fn, input):
        output = fn(input)
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        return (None,) + out_relevance


class epsilon_lrp(Function):

    @staticmethod
    def forward(ctx, fn, epsilon, *inputs):
        # create boolean mask for inputs requiring gradients
        requires_grads = [True if inp.requires_grad else False for inp in inputs]
        if sum(requires_grads) == 0:
            # no gradients to compute or gradient checkpointing is used
            return fn(*inputs)

        # detach inputs to avoid overwriting gradients if same input is used as multiple arguments (like in self-attention)
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        with torch.enable_grad():
            outputs = fn(*inputs)

        ctx.epsilon, ctx.requires_grads = epsilon, requires_grads
        # save only inputs requiring gradients
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, outputs)
        return outputs.detach()

    @staticmethod
    def backward(ctx, *out_relevance):
        inputs, outputs = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        relevance_norm = out_relevance[0] / _stabilize(outputs, ctx.epsilon, inplace=False)

        # computes vector-jacobian product
        grads = torch.autograd.grad(outputs, inputs, relevance_norm)

        # return relevance at requires_grad indices else None
        relevance = iter([grads[i].mul_(inputs[i]) for i in range(len(inputs))])
        return (None, None) + tuple(next(relevance) if req_grad else None for req_grad in ctx.requires_grads)


class add_tensors_lrp(Function):
    """
    Target to residual connection
    """
    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False, epsilon=1e-6, tag=None):
        outputs = input_a + input_b

        ctx.save_for_backward(input_a, input_b, outputs)
        ctx.epsilon, ctx.inplace = epsilon, inplace
        ctx.diag_tag = tag

        return outputs

    @staticmethod
    def backward(ctx, *out_relevance):
        input_a, input_b, outputs = ctx.saved_tensors

        diag_on = diag.lrp_active()
        raw_outputs = outputs.detach().clone() if (diag_on and ctx.inplace) else outputs
        denominator = _stabilize(outputs, epsilon=ctx.epsilon, inplace=ctx.inplace)
        relevance_norm = out_relevance[0] / denominator

        relevance_a = relevance_norm * input_a
        relevance_b = relevance_norm * input_b
        relevance_a_raw, relevance_b_raw = relevance_a, relevance_b

        relevance_a = torch.nan_to_num(relevance_a, nan=0.0, posinf=1e12, neginf=-1e12)
        relevance_b = torch.nan_to_num(relevance_b, nan=0.0, posinf=1e12, neginf=-1e12)

        if diag_on:
            diag_relevance.residual_add(ctx.diag_tag, input_a, input_b, raw_outputs, out_relevance[0], denominator,
                                        relevance_a_raw, relevance_b_raw, relevance_a, relevance_b)

        return relevance_a, relevance_b, None, None, None


class layer_norm_lrp(Function):
    """
    Identity Rule
    """
    @staticmethod
    def forward(ctx, x, weight, bias, variance_epsilon, epsilon=1e-6):

        with torch.enable_grad():
            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            std = (var + variance_epsilon).sqrt()
            y = (x - mean) / std.detach() # detach std operation will remove it from computational graph i.e. identity rule on x/std

            if weight is not None:
                y *= weight
            if bias is not None:
                y += bias
            ctx.save_for_backward(x, y)
            ctx.epsilon = epsilon

        return y.detach()

    @staticmethod
    def backward(ctx, *out_relevance):
        return *out_relevance ,None, None, None


class filter_lrp(Function):
    @staticmethod
    def forward(ctx, input, top_k_percent, tag=None):
        ctx.top_k_percent = top_k_percent
        ctx.diag_tag = tag
        # 診断用: Filter の forward 入力（= 次に関連度を戻す演算の分母）への参照。計算には使わない
        ctx.diag_input = input.detach() if diag.lrp_active() else None
        return input

    @staticmethod
    def backward(ctx, *out_relevance):
        top_k_percent = ctx.top_k_percent
        out_relevance = out_relevance[0]

        assert 0 <= top_k_percent <= 1

        if 0 < top_k_percent < 1.0:
            size = out_relevance.size()
            # CNN:[B,C,H,W]
            if len(size) == 4:
                relevance = out_relevance.reshape(-1) #[B,C*H*W]
                original_sum = torch.sum(torch.abs(out_relevance), dim=(2, 3), keepdim=True) # [B, C, 1, 1]

                num_elements = relevance.size(-1)
                k = max(1, int(top_k_percent * num_elements))
                top_k = torch.topk(input=torch.abs(relevance), k=k, dim=-1)

                mask = torch.zeros_like(relevance, dtype=torch.bool)
                mask.scatter_(dim=-1, index=top_k.indices, value=True)

                relevance = torch.where(mask, relevance, torch.tensor(0.0, device=relevance.device))
                relevance = relevance.view(size)
                masked = relevance

                selected_sum = torch.sum(torch.abs(relevance), dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
                scale = original_sum / _stabilize(selected_sum)

                relevance = relevance * scale # Normalize per channel
                rescaled_raw = relevance
                relevance = torch.nan_to_num(relevance, nan=0.0, posinf=1e12, neginf=-1e12)

                if diag.lrp_active():
                    diag_relevance.filter_cnn(ctx.diag_tag, top_k_percent, out_relevance, mask, masked,
                                              original_sum, selected_sum, scale, rescaled_raw, relevance,
                                              ctx.diag_input)

                return relevance, None, None

            # Transformer : [B, N, C]
            elif len(size) == 3:
                relevance = out_relevance.flatten(start_dim=1)
                original_sum = torch.sum(relevance, dim=1, keepdim=True)

                num_elements = relevance.size(-1)
                k = max(1, int(top_k_percent * num_elements))
                top_k = torch.topk(input=torch.abs(relevance), k=k, dim=-1)
                relevance = torch.zeros_like(relevance)
                relevance.scatter_(dim=1, index=top_k.indices, src=out_relevance.view(size[0], -1).gather(dim=1, index=top_k.indices))

                selected_sum = torch.sum(relevance, dim=1, keepdim=True)
                masked = relevance
                relevance = relevance * (original_sum / selected_sum)

                if diag.lrp_active():
                    diag_relevance.filter_transformer(ctx.diag_tag, top_k_percent, out_relevance, top_k.indices,
                                                      masked.view(size), original_sum, selected_sum,
                                                      relevance.view(size), ctx.diag_input)

                return relevance.view(size), None, None

        elif top_k_percent == 0.0:
            relevance = torch.zeros_like(out_relevance)
            if diag.lrp_active():
                diag_relevance.gate(ctx.diag_tag, top_k_percent, out_relevance, relevance)
            return relevance, None, None

        else:
            if diag.lrp_active():
                diag_relevance.passthrough_filter(ctx.diag_tag, top_k_percent, out_relevance)
            return out_relevance, None, None


class epsilon_conv2d_lrp(Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, stride, padding, dilation, epsilon=1e-9, tag=None):
        outputs = F.conv2d(inputs, weight, bias, stride, padding, dilation)
        ctx.save_for_backward(inputs, weight, bias, outputs)
        ctx.epsilon = epsilon
        ctx.diag_tag = tag
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        return outputs

    @staticmethod
    def backward(ctx, *out_relevance):
        inputs, weight, bias, outputs = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        out_relevance = out_relevance[0]
        out_relevance_raw = out_relevance

        z = _stabilize(outputs, epsilon=ctx.epsilon)
        z = torch.where(z > 0, torch.clamp(z, min=ctx.epsilon), torch.clamp(z, max=-ctx.epsilon) )

        out_relevance = out_relevance / z

        z_grad = F.conv_transpose2d(
            out_relevance,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation)


        if z_grad.size(-2) > inputs.size(-2) or z_grad.size(-1) > inputs.size(-1):
            # Crop if z_grad is too large
            z_grad = z_grad[..., :inputs.size(-2), :inputs.size(-1)]
        elif z_grad.size(-2) < inputs.size(-2) or z_grad.size(-1) < inputs.size(-1):
            # Pad if z_grad is too small
            pad_h = inputs.size(-2) - z_grad.size(-2)
            pad_w = inputs.size(-1) - z_grad.size(-1)
            z_grad = F.pad(z_grad, (0, pad_w, 0, pad_h))

        relevance = z_grad * inputs
        relevance_raw = relevance
        relevance = torch.nan_to_num(relevance, nan=0.0, posinf=1e12, neginf=-1e12)

        if diag.lrp_active():
            diag_relevance.conv2d(ctx.diag_tag, inputs, outputs, z, out_relevance_raw, relevance_raw, relevance)

        return relevance, None, None, None, None, None, None, None


class avgpool2d_lrp(Function):
    @staticmethod
    def forward(ctx, inputs, output_size):
        ctx.save_for_backward(inputs)
        outputs = F.adaptive_avg_pool2d(inputs, output_size)
        size = outputs.shape
        ctx.output_size = size

        return outputs

    @staticmethod
    def backward(ctx, *out_relevance):
        inputs, = ctx.saved_tensors
        output_size = ctx.output_size
        B, C, H_in, W_in = inputs.shape
        _, _, H_out, W_out = output_size


        stride_h = math.floor(H_in / H_out)
        stride_w = math.floor(W_in / W_out)

        kernel_size_h = H_in - (H_out - 1) * stride_h
        kernel_size_w = W_in - (W_out - 1) * stride_w

        weight = torch.ones((C, 1, kernel_size_h, kernel_size_w), device=out_relevance[0].device) / (kernel_size_h * kernel_size_w)

        relevance = F.conv_transpose2d(
            out_relevance[0], weight, stride=(stride_h, stride_w), padding=0, output_padding=0, groups=C
        )

        if torch.isnan(relevance).any():
            print("Found NaN in intermediate result")

        return relevance, None


class maxpool2d_lrp(Function):
    @staticmethod
    def forward(ctx, inputs, kernel_size, stride=None, padding=0):
        outputs, indices = F.max_pool2d(
            inputs, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
        ctx.save_for_backward(inputs, indices)
        ctx.kernel_size = kernel_size
        ctx.stride = stride or kernel_size
        ctx.padding = padding
        return outputs

    @staticmethod
    def backward(ctx, out_relevance):

        inputs, indices = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding

        relevance = F.max_unpool2d(out_relevance, indices, kernel_size=kernel_size, stride=stride, padding=padding, output_size=inputs.size())

        return relevance, None, None, None


class layernorm_lrp(Function):
    """
    A mixture of identity and epsilon rules for standard nn.LayerNorm operations:
    Idenitiy rule for element-wise (y * weight) because single input single output. Identity rule on 1/std because of Proposition 3.4 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'.
    (x - mean) is a linear operation, so we apply the epsilon rule on it.

    To implement this, we do a trick: We differentiate the whole layer, while detaching the std operation from the graph.
    This is then equivalent to all the rules discussed above! This is slightly faster than implementing everything in pure lxt.

    Parameters:
    -----------
    hidden_states: torch.Tensor
        The input tensor
    weight: torch.Tensor
        The weight tensor
    variance_epsilon: float
        Small value to stabilize the denominator
    """

    @staticmethod
    def forward(ctx, x, weight, bias, variance_epsilon, epsilon=1e-6):

        with torch.enable_grad():

            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            std = (var + variance_epsilon).sqrt()
            y = (x - mean) / std.detach() # detach std operation will remove it from computational graph i.e. identity rule on x/std
            if weight is not None:
                y *= weight
            if bias is not None:
                y += bias

            ctx.save_for_backward(x, y)
            ctx.epsilon = epsilon

        return y.detach()

    @staticmethod
    def backward(ctx, *out_relevance):

        x, y = ctx.saved_tensors

        relevance_norm = out_relevance[0] / _stabilize(y, ctx.epsilon, False)

        grads, = torch.autograd.grad(y, x, relevance_norm)

        return (grads*x, None, None, None, None)


