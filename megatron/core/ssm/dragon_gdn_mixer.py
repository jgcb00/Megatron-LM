# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.dragon_config import DragonConfig
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

try:
    from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
    from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                      fused_recurrent_gated_delta_rule)
except ImportError:
    raise ImportError("fla is required by the Gated DeltaNet model but cannot be imported")

try:
    from einops import rearrange
except ImportError:
    raise ImportError("einops is required by the Mamba model but cannot be imported")


@dataclass
class DragonGDNMixerSubmodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = None


class DragonGDNMixer(MegatronModule):
    """
    Args:
        config: The config of the model.
        submodules: Contains the module specs for the input and output linear layers.
        d_model: The hidden size of the model.
        d_conv: The number of channels in the causal convolution.
        expand_v: The expansion factor for the values.
        d_head: The hidden size of each attention head.
        bias: Whether to use bias in the linear layers.
        conv_bias: Whether to use bias in the causal convolution.
        conv_init: The initialization range for the causal convolution weights.
        layer_number: The layer number of this Mamba layer.
    """

    def __init__(
        self,
        config: DragonConfig,
        submodules: DragonGDNMixerSubmodules,
        d_model,
        d_conv=4,
        expand_v=2,
        d_head=64,
        conv_bias=False,
        conv_init=None,
        layer_number=None,
    ):
        super().__init__(config)
        self.config = config
        
        self.d_model = d_model
        self.expand_v = expand_v

        self.conv_size = d_conv
        self.conv_bias = conv_bias
        self.conv_init = conv_init

        self.head_dim = d_head
        self.n_heads = d_model // d_head

        self.key_dim = self.n_heads * self.head_dim
        self.value_dim = self.key_dim * self.expand_v
        self.head_k_dim = d_head
        self.head_v_dim = d_head * self.expand_v
        self.layer_idx = layer_number
        self.silu = nn.SiLU()

        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()

        print("Tensor Model Parallel Size: ", self.tensor_model_parallel_size)

        self.n_heads_local = self.n_heads // self.tensor_model_parallel_size

        # TODO (TP) : key_dim, value_dim etc should be local!!

        in_proj_dim = (
            self.key_dim +  # q_proj
            self.key_dim +  # k_proj
            self.value_dim +  # v_proj
            self.n_heads +  # b_proj
            self.n_heads  # a_proj
        )

        self.q_slice = slice(0, self.key_dim)
        self.k_slice = slice(self.key_dim, 2 * self.key_dim)
        self.v_slice = slice(2 * self.key_dim, 2 * self.key_dim + self.value_dim)
        self.b_slice = slice(
            2 * self.key_dim + self.value_dim,
            2 * self.key_dim + self.value_dim + self.n_heads,
        )
        self.a_slice = slice(
            2 * self.key_dim + self.value_dim + self.n_heads,
            2 * self.key_dim + self.value_dim + 2 * self.n_heads,
        )

        self.in_proj = build_module(
            submodules.in_proj,
            self.config,
            d_model,
            in_proj_dim,
            bias=False,
            skip_bias_add=False,
            init_method=self.config.init_method,
        )

        # hard coded for now todo
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        A_init_range=(1, 16)

        with get_cuda_rng_tracker().fork():
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(
                    self.n_heads_local, device=torch.cuda.current_device(), dtype=config.params_dtype
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_bias = nn.Parameter(inv_dt)
            # Our initialization would set all Linear.bias to zero,
            # need to mark this one as _no_reinit
            self.dt_bias._no_reinit = True
            # Just to be explicit. Without this we already don't
            # put wd on dt_bias because of the check

            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True

            assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
            A = torch.empty(
                self.n_heads_local, dtype=torch.float32, device=torch.cuda.current_device()
            ).uniform_(*A_init_range)
            A_log = torch.log(A)  # Keep A_log in fp32
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            setattr(self.A_log, 'tensor_model_parallel', True)

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(
                self.n_heads_local,
                device=torch.cuda.current_device(),
            )
        )  # Keep in fp32
        self.D._no_weight_decay = True
        setattr(self.D, 'tensor_model_parallel', True)

        with get_cuda_rng_tracker().fork():
            # ShortConvolution is a wrapper around nn.Conv1d (for definition) and causal_conv1d (for forward)
            self.q_conv1d = ShortConvolution(
                    hidden_size=self.key_dim,
                    kernel_size=self.conv_size,
                    activation='silu'
                )
            self.k_conv1d = ShortConvolution(
                    hidden_size=self.key_dim,
                    kernel_size=self.conv_size,
                    activation='silu'
                )
            self.v_conv1d = ShortConvolution(
                    hidden_size=self.value_dim,
                    kernel_size=self.conv_size,
                    activation='silu'
                )
            
            setattr(self.q_conv1d.weight, 'tensor_model_parallel', True)
            setattr(self.q_conv1d.bias, 'tensor_model_parallel', True)

            setattr(self.k_conv1d.weight, 'tensor_model_parallel', True)
            setattr(self.k_conv1d.bias, 'tensor_model_parallel', True)

            setattr(self.v_conv1d.weight, 'tensor_model_parallel', True)
            setattr(self.v_conv1d.bias, 'tensor_model_parallel', True)

            if self.conv_init is not None:
                nn.init.uniform_(self.q_conv1d.weight, -self.conv_init, self.conv_init)
                nn.init.uniform_(self.k_conv1d.weight, -self.conv_init, self.conv_init)
                nn.init.uniform_(self.v_conv1d.weight, -self.conv_init, self.conv_init)
        
        #self.apply(self._initialize_weights)
        # original GDN used a xavier_uniform here for the nn.Linear weights
        # I disabled it because we instead use the self.config.init_method

    def forward(self, hidden_states):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """
        _, batch, dim = hidden_states.shape

        qkvba, _ = self.in_proj(hidden_states)

        # transpose: l b pd --> b l pd
        qkvba = rearrange(qkvba, "l b d -> b l d").contiguous()
        
        # split proj into q, k, v, b, a
        q_proj = qkvba[:, :, self.q_slice]
        k_proj = qkvba[:, :, self.k_slice]
        v_proj = qkvba[:, :, self.v_slice]
        b_proj = qkvba[:, :, self.b_slice]
        a_proj = qkvba[:, :, self.a_slice]

        q, _ = self.q_conv1d(x=q_proj,
                             mask=None, 
                             cache=None,
                             output_final_state=False,
                             seq_idx=None)
        k, _ = self.k_conv1d(x=k_proj,
                             mask=None,
                             cache=None,
                             output_final_state=False,
                             seq_idx=None)
        v, _ = self.v_conv1d(x=v_proj,
                             mask=None,
                             cache=None,
                             output_final_state=False,
                             seq_idx=None)
        
        q, k = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        beta = b_proj.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a_proj.float() + self.dt_bias)

        o, _ = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=None, # for varlen training
                head_first=False,
                use_qk_l2norm_in_kernel=True
            ) # (b t h d) where d is head_v_dim
        
        # here skip output norm and out_proj as we are in a Hymba-like setup
        
        o = rearrange(o, 'b t h d -> t b (h d)').contiguous()
        return o
    
    # todo
    #def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):