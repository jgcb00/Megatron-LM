# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.jit import jit_fuser

# BIAS RELU SQUARED FUSION/ NO AUTOGRAD ################


@jit_fuser
def bias_relu_squared(bias, y):
    x = bias + y
    return torch.relu(x) ** 2 


# gradient of rrelu_squared
@jit_fuser
def bias_relu_squared_back(g, bias, y):
    x = bias + y
    ff = 2 * torch.relu(x)
    return ff * g


class ReLUSquaredFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_relu_squared(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_relu_squared(grad_output, bias, input)
        return tmp, tmp

    # This is required to make Sphinx happy :-(
    @classmethod
    def apply(cls, *args, **kwargs):
        return super().apply(*args, **kwargs)


bias_relu2_impl = ReLUSquaredFunction.apply