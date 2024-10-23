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
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from .fast_mlp_visualisation import fffn2picture
import time
import fastfeedforward_cuda

@dataclass
class FastMLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None
    parallel_trees: Optional[int] = 128
    master_node: Optional[bool] = True
    master_node_width: Optional[int] = None
    load_balancing_update_rate: Optional[float] = 1e-3


@dataclass
class FFFN_config:
    parallel_trees: int
    hidden_size: int
    load_balancing_update_rate: float
    tensor_model_parallel_size: int
    master_node : bool = True
    master_node_width: int = None


    def __post_init__(self):
        assert (
            self.parallel_trees % self.tensor_model_parallel_size == 0
        ), "FFFN Tree can't be divided between gpu"
        if self.master_node and self.master_node_width is None:
            self.master_node_width = (
                ceil(self.depth / self.parallel_trees) * self.parallel_trees
            )
        elif self.master_node:
            self.master_node_width = (
                ceil(self.master_node_width / self.parallel_trees) * self.parallel_trees
            )
        else :
            self.master_node_width = 0
        self.hidden_size = self.n_nodes * self.parallel_trees
    
    @property
    def n_nodes(self):
        return 2**self.depth - 1 + self.master_node_width_by_parallel_tree 
    
    @property
    def depth(self) -> int:
      return int(ceil(log2(self.hidden_size / self.parallel_trees)))
    
    @property
    def master_node_width_by_parallel_tree(self) -> int:
        return int(self.master_node_width / self.parallel_trees)
    
    @property
    def parallel_trees_by_gpu(self) -> int:
        return int(self.parallel_trees / self.tensor_model_parallel_size)
    
    


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
        self.config: TransformerConfig = config
        self.input_size = input_size if input_size != None else self.config.hidden_size
        self.fffn_config = FFFN_config(
            parallel_trees=submodules.parallel_trees,
            hidden_size=self.input_size*4,
            load_balancing_update_rate=submodules.load_balancing_update_rate,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
        )
        self.linear_fc1 = torch.nn.Parameter(torch.randn(
            (self.input_size, self.fffn_config.hidden_size), dtype=torch.bfloat16
        ))
        self.bias = torch.nn.Parameter(torch.zeros((self.fffn_config.hidden_size,), dtype=torch.bfloat16))
        self.linear_fc2 = torch.nn.Parameter(torch.randn(
            (self.fffn_config.hidden_size, self.input_size), dtype=torch.bfloat16
        ))
        
        #Load balancing bias
        self.lb_bias = torch.nn.Parameter(torch.zeros((self.fffn_config.hidden_size,)), dtype=torch.bfloat16, requires_grad=False)
        self.load = None
        self.work = None
        left_children = torch.tensor(
            list(
                itertools.chain.from_iterable(
                    [((2 ** d - 1) + n) * 2 + 1 for n in range(2**d)]
                    for d in range(self.fffn_config.depth - 1)
                )
            )
        ).unsqueeze(0).expand(self.fffn_config.parallel_trees, -1) + (torch.arange(self.fffn_config.parallel_trees) * ((2**self.fffn_config.depth -1) + self.fffn_config.master_node_width_by_parallel_tree) ).unsqueeze(1)

        right_children = left_children + 1
        leave_fake_children = torch.zeros((self.parallel_trees, 2 ** (self.fffn_config.depth - 1) + self.fffn_config.master_node_width_by_parallel_tree), dtype=torch.long)

        self.left_children = torch.cat([left_children, leave_fake_children], dim=1).view(-1)
        self.right_children = torch.cat([right_children, leave_fake_children], dim=1).view(-1)
        
        #Visualisation:
        self.visualisation = False
        self.eval_started = False
        self.nb_tokens = 0



    def forward(self, hidden_states):
        # Here we take the assumptions of the leonardo booster node that have 4 GPUs
        # Meaning we will try one binary tree per GPU
        if self.work is not None:
            self.work.wait()
            # self.update_sign[(self.update_sign > -5) & (self.update_sign < 5)] = 0
            self.load = torch.clamp(self.load, min=-1, max=1)
            self.lb_bias.data = self.lb_bias.data + self.fffn_config.load_balancing_update_rate * self.update_sign
            self.work = None
            self.load = None
    
        output, activated_nodes = fastfeedforward_cuda.fffn_function(
            hidden_states,
            self.linear_fc1,
            self.bias,
            self.linear_fc2,
            self.lb_bias,
            self.input_size,
            self.fffn_config.depth,
            self.fffn_config.parallel_trees,
            self.fffn_config.master_node_width_by_parallel_tree,
            self.fffn_config.n_nodes,
        )
    
        node, load = torch.unique(activated_nodes.view(-1))
        total_load = torch.zeros(self.fffn_config.hidden_size, dtype=torch.int32)
        total_load[node] = load
        
        if self.training and self.work is None:
            self.load = total_load[self.left_children] - total_load[self.right_children]
            self.work = dist.all_reduce(self.load, op=dist.ReduceOp.SUM, async_op=True)
        
        if self.visualisation:
            self.visualise(total_load, hidden_states.size(0)*hidden_states.size(1))
                    
        return output, None

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict
    
    def visualise(self, total_load, batch_size):
       
        if self.eval_started is False and not self.training:
            self.eval_started = True
            self.cum_load = torch.zeros_like(self.fffn_config.hidden_size)
            
        if not self.training and self.eval_started:
            self.cum_load += total_load
            self.nb_tokens += batch_size

        if self.eval_started and self.training:
            #Meaning end of the evaluation
            dist.all_reduce(self.cum_load, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0:
                self.cum_load = self.cum_load.view(self.parallel_trees, -1)
                print(self.cum_load)
                world_size = dist.get_world_size()
                self.nb_tokens = self.nb_tokens * world_size
                with torch.no_grad():
                    fffn2picture(
                        self.update_sign,
                        self.nb_tokens,
                        self.parallel_trees_by_gpu,
                        self.master_node_width_by_parallel_tree,
                        hash(self),
                    )
            self.nb_tokens = 0
            self.eval_started = False
            