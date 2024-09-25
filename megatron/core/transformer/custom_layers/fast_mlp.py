# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from math import log2, ceil
import numpy as np
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.training import get_args

@dataclass
class FastMLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None
    parallel_trees: Optional[int] = 4
    master_node: Optional[bool] = True
    master_node_width: Optional[int] = None


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
        tensor_model_parallel_size = args.tensor_model_parallel_size
        assert submodules.parallel_trees % tensor_model_parallel_size == 0, "FFFN Tree can't be divided between gpu"
        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size
        
        depth = int(ceil(log2(self.config.ffn_hidden_size/submodules.parallel_trees)))
        
        print(f"Depth: {depth}")
        if submodules.master_node and submodules.master_node_width is None:
            #it has to be a multiple of tensor_model_parallel_size, to avoid issue with tensor model parallelism
            submodules.master_node_width = ceil(depth / submodules.parallel_trees) * submodules.parallel_trees
        elif submodules.master_node_width is None:
            submodules.master_node_width = 0
        
        #The fused kernel multiplies the hidden size by 4, so we need to divide by 4
        ffn_hidden_size = int((2**depth - 1) * submodules.parallel_trees + submodules.master_node_width)
        print(f"FFN Hidden Size: {ffn_hidden_size}")
        
        # if self.config.gated_linear_unit:
        #     ffn_hidden_size *= 2
        # We divide by four as each gpu will activate a part of the hiddensize
        self.master_node_width = int(submodules.master_node_width) 
        self.master_node_width_by_parallel_tree = int(submodules.master_node_width / submodules.parallel_trees)
        print(f"Master Node Width: {self.master_node_width}")
        self.parallel_trees = submodules.parallel_trees
        print(f"Parallel Trees: {self.parallel_trees}")
        self.parallel_trees_by_gpu = int(submodules.parallel_trees / tensor_model_parallel_size)
        self.depth = depth

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method, #will probably have to update this
            gather_output=False,
            bias=True, #self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert, #false
            tp_comm_buffer_name='fc1',
        )

        self.activation_func = self.config.activation_func #should be Gelu() F.gelu

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            ffn_hidden_size,
            self.input_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False, #self.config.add_bias_linear,
            input_is_parallel=True, # Probably should be False
            skip_bias_add=True,
            is_expert=is_expert, #False
            tp_comm_buffer_name='fc2',
        )

    def forward(self, hidden_states):
        #Here we take the assumptions of the leonardo booster node that have 4 GPUs
        # Meaning we will try one binary tree per GPU

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        intermediate_parallel = apply_custom_fff_activation(
            intermediate_parallel, 
            bias_parallel, 
            self.master_node_width_by_parallel_tree, 
            self.parallel_trees_by_gpu, 
            self.depth,
        )
        
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

def apply_custom_fff_activation(intermediate_parallel, bias_parallel, master_node_width, parallel_trees, depth):
    
    flatten_intermediate = intermediate_parallel.view(-1, intermediate_parallel.size(-1))
    
    logit_decisions = (flatten_intermediate > 0).long() # (batch_size, parallel_size * n_nodes + master_node_size)
    logit_decisions = logit_decisions.view(-1, parallel_trees, 2**depth-1 + master_node_width) # (batch_size, parallel_size, n_nodes)
    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
    
    batch_size = flatten_intermediate.size(0)

    decisions = logit_decisions.view(batch_size, parallel_trees, -1) # (batch_size, parallel_size, n_nodes)
    print("Decisions shape:", decisions.shape)
    with torch.no_grad():
        current_nodes = torch.zeros((batch_size, parallel_trees), dtype=torch.long, device=intermediate_parallel.device)
        decision_map = torch.zeros_like(decisions, dtype=torch.float, device=intermediate_parallel.device) # (batch_size, parallel_size, n_nodes)
        decision_map.scatter_(dim=2, index=current_nodes.unsqueeze(-1), value=1.0) # set the first node to 1
        for d in range(depth):
            current_platform = 2 ** d - 1
            next_platform = 2 ** (d + 1) - 1
            moves = torch.gather(decisions, 2, current_nodes.unsqueeze(2)).squeeze(2)
            next_nodes = (current_nodes - current_platform) * 2 + moves + next_platform
            decision_map.scatter_(2, next_nodes.unsqueeze(-1), 1.0)
            current_nodes = next_nodes
        decision_map[:, :, master_node_width:] = 1.0
        decision_map = decision_map.flatten(1,2)
    print("Intermediate Parallel shape:", intermediate_parallel.shape)
    intermediate_parallel =  intermediate_parallel * decision_map
    return intermediate_parallel.view(intermediate_parallel.size(0), intermediate_parallel.size(1), -1)