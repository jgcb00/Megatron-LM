# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
import itertools
from typing import Optional, Tuple, Union
from math import log2, ceil, sqrt
import torch
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedStateDict
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from .fast_mlp_visualisation import fffn2picture
import time

@dataclass
class FastMLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None
    parallel_trees: Optional[int] = 4
    master_node: Optional[bool] = True
    master_node_width: Optional[int] = None
    load_balancing_update_rate: Optional[float] = 1e-4


class FastMLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: FastMLPSubmodules,
        is_expert: bool = False,
        input_size: int = None,
    ):
        super().__init__(config=config)
        args = get_args()
        self.visualisation = False
        tensor_model_parallel_size = args.tensor_model_parallel_size
        assert (
            submodules.parallel_trees % tensor_model_parallel_size == 0
        ), "FFFN Tree can't be divided between gpu"
        self.config: TransformerConfig = config
        self.input_size = input_size if input_size != None else self.config.hidden_size

        depth = int(ceil(log2(self.config.ffn_hidden_size / submodules.parallel_trees)))

        if submodules.master_node and submodules.master_node_width is None:
            # it has to be a multiple of tensor_model_parallel_size, to avoid issue with tensor model parallelism
            submodules.master_node_width = (
                ceil(depth / submodules.parallel_trees) * submodules.parallel_trees
            )
        elif submodules.master_node_width is None:
            submodules.master_node_width = 0

        # The fused kernel multiplies the hidden size by 4, so we need to divide by 4
        ffn_hidden_size = int(
            (2**depth - 1) * submodules.parallel_trees + submodules.master_node_width
        )

        self.master_node_width = int(submodules.master_node_width)
        self.master_node_width_by_parallel_tree = int(
            submodules.master_node_width / submodules.parallel_trees
        )

        self.parallel_trees = submodules.parallel_trees

        self.parallel_trees_by_gpu = int(submodules.parallel_trees / tensor_model_parallel_size)
        self.depth = depth

        init_k = sqrt(1.0 / self.input_size)
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=partial(
                torch.nn.init.uniform_, a=-init_k, b=init_k
            ),  # will probably have to update this
            gather_output=False,
            bias=True,  # self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,  # false
            tp_comm_buffer_name='fc1',
        )
        if self.visualisation:
            self.usage = torch.zeros(ffn_hidden_size, dtype=torch.int32, device='cuda')
            self.nb_tokens = 0
            self.threshold = 1_000_000
        self.activation_func = self.config.activation_func  # should be Gelu() F.gelu

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            ffn_hidden_size,
            self.input_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,  # self.config.add_bias_linear,
            input_is_parallel=True,  # Probably should be False
            skip_bias_add=True,
            is_expert=is_expert,  # False
            tp_comm_buffer_name='fc2',
        )

        self.update_rate = submodules.load_balancing_update_rate
        self.lb_bias = torch.nn.Parameter(torch.zeros((ffn_hidden_size,)), requires_grad=False)
        self.update_sign = None
        self.work = None
        left_children = torch.tensor(
            list(
                itertools.chain.from_iterable(
                    [((2 ** d - 1) + n) * 2 + 1 for n in range(2**d)]
                    for d in range(depth - 1)
                )
            )
        ).unsqueeze(0).expand(self.parallel_trees, -1) + (torch.arange(self.parallel_trees) * ((2**depth -1) + self.master_node_width_by_parallel_tree) ).unsqueeze(1)

        right_children = left_children + 1


        leave_fake_children = torch.zeros((self.parallel_trees, 2 ** (depth - 1) + self.master_node_width_by_parallel_tree), dtype=torch.long)

        self.left_children = torch.cat([left_children, leave_fake_children], dim=1).view(-1)
        self.right_children = torch.cat([right_children, leave_fake_children], dim=1).view(-1)


    def forward(self, hidden_states):
        # Here we take the assumptions of the leonardo booster node that have 4 GPUs
        # Meaning we will try one binary tree per GPU
        if self.work is not None:
            self.work.wait()
            self.update_sign[(self.update_sign > -50) & (self.update_sign < 50)] = 0
            self.update_sign = -torch.clamp(self.update_sign, min=-1, max=1)
            self.lb_bias.data = self.lb_bias.data + self.update_rate * self.update_sign
            self.work = None
    
        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        intermediate_parallel, cum_decision_map = apply_custom_fff_activation(
            intermediate_parallel,
            bias_parallel,
            self.lb_bias,
            self.master_node_width_by_parallel_tree,
            self.parallel_trees_by_gpu,
            self.depth,
        )
        if self.training and self.work is None:
            self.update_sign = cum_decision_map[self.left_children] - cum_decision_map[self.right_children]
            self.work = dist.all_reduce(self.update_sign, op=dist.ReduceOp.SUM, async_op=True)
        if self.visualisation:
            with torch.no_grad():
                self.usage.to(mask.device)
                print(mask)
                self.usage += mask
                self.nb_tokens += hidden_states.size(0) * hidden_states.size(1)
                if self.nb_tokens > self.threshold:
                    self.threshold += 200_000_000
                    fffn2picture(
                        self.usage,
                        self.nb_tokens,
                        self.parallel_trees_by_gpu,
                        self.master_node_width_by_parallel_tree,
                        hash(self),
                    )
                    self.usage.zero_()
                    self.nb_tokens = 0
        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict


def apply_custom_fff_activation(
    intermediate_parallel, bias_parallel, load_balancing_bias, master_node_width, parallel_trees, depth
):

    flatten_intermediate = intermediate_parallel.view(-1, intermediate_parallel.size(-1))
    if bias_parallel is not None:
        logit_decisions = (
            (flatten_intermediate + bias_parallel + load_balancing_bias.unsqueeze(0)) > 0
        ).long()  # (batch_size, parallel_size * n_nodes + master_node_size)
    else:
        logit_decisions = ((flatten_intermediate + load_balancing_bias.unsqueeze(0)) > 0).long()

    logit_decisions = logit_decisions.view(
        -1, parallel_trees, 2**depth - 1 + master_node_width
    )  # (batch_size, parallel_size, n_nodes)
    if bias_parallel is not None:
        flatten_intermediate = bias_gelu_impl(flatten_intermediate, bias_parallel)
    else:
        flatten_intermediate = bias_gelu_impl(flatten_intermediate, torch.zeros((flatten_intermediate.shape[-1]), device=flatten_intermediate.device, dtype=flatten_intermediate.dtype))
    batch_size = flatten_intermediate.size(0)

    decisions = logit_decisions.view(
        batch_size, parallel_trees, -1
    )  # (batch_size, parallel_size, n_nodes)
    with torch.no_grad():
        current_nodes = torch.zeros(
            (batch_size, parallel_trees), dtype=torch.long, device=intermediate_parallel.device
        )
        decision_map = torch.zeros_like(
            decisions, dtype=torch.bfloat16, device=intermediate_parallel.device
        )  # (batch_size, parallel_size, n_nodes)
        decision_map.scatter_(
            dim=2, index=current_nodes.unsqueeze(-1), value=1.0
        )  # set the first node to 1
        for d in range(depth - 1):
            current_platform = 2**d - 1
            next_platform = 2 ** (d + 1) - 1
            moves = torch.gather(decisions, 2, current_nodes.unsqueeze(2)).squeeze(2)
            next_nodes = (current_nodes - current_platform) * 2 + moves + next_platform
            decision_map.scatter_(2, next_nodes.unsqueeze(-1), 1.0)
            current_nodes = next_nodes
        decision_map[:, :, -master_node_width:] = 1.0
        cum_sum = decision_map.view(batch_size, -1).sum(dim=0)
        decision_map = decision_map.flatten(1, 2)

    flatten_intermediate = flatten_intermediate * decision_map
    return (
        flatten_intermediate.view(intermediate_parallel.size(0), intermediate_parallel.size(1), -1),
        cum_sum,
    )
