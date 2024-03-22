# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Model and data parallel groups."""

import torch
from megatron import get_args
from .utils import ensure_divisibility
#[JK]

import json

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage
_PIPELINE_GLOBAL_RANKS = None
_MODEL_PARALLEL_GLOBAL_GROUP = None

def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


            
def initialize_model_parallel(tensor_model_parallel_size_=1,
                              pipeline_model_parallel_size_=1,
                              virtual_pipeline_model_parallel_size_=None):
    
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used to parallelize model tensor.
        pipeline_model_parallel_size: number of GPUs used to parallelize model pipeline.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    args = get_args()
    if torch.distributed.get_rank() == 0:
        print('> initializing tensor model parallel with size {}'.format(
            tensor_model_parallel_size_))
        print('> initializing pipeline model parallel with size {}'.format(
            pipeline_model_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size)
    ensure_divisibility(world_size,
                        tensor_model_parallel_size * pipeline_model_parallel_size)
    data_parallel_size = world_size // (tensor_model_parallel_size *
                                        pipeline_model_parallel_size)

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    num_data_parallel_groups = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size_ is not None:
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size_

    rank = torch.distributed.get_rank()

    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
            'tensor model parallel group is already initialized'
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
            'pipeline model parallel group is already initialized' 
    global _EMBEDDING_GROUP
    assert _EMBEDDING_GROUP is None, \
            'embedding group is already initialized' 
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
            'data parallel group is already initialized'  
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
            'model parallel group is already initialized' 
            
    global _MODEL_PARALLEL_GLOBAL_GROUP
    assert _MODEL_PARALLEL_GLOBAL_GROUP is None, \
            'model parallel global rank is already initialized'  
    
    if args.topo_aware:
    
        with open(args.assignment_json_file, "r") as f:
            opt_pp_sequence_data = json.load(f)
        
        opt_ranking_num = args.opt_ranking_num
        # if args.pipeline_ranking == "middle":
        #     opt_ranking_num = len(opt_pp_sequence_data) // 2
        # elif args.pipeline_ranking == "last":
        #     opt_ranking_num = len(opt_pp_sequence_data)-1
        
        len_opt_pp = len(opt_pp_sequence_data[opt_ranking_num][0])
        len_opt_pp_element = len(opt_pp_sequence_data[opt_ranking_num][0][0])
        
        
        print(f"search pp sequence length : {len_opt_pp}")
        print(f"opt_pp_sequence_data : {opt_pp_sequence_data[opt_ranking_num][0]}")
        print(f"len_opt_pp_element : {len_opt_pp_element}")  
        print(f"ranking of opt_pp_sequence {opt_ranking_num}")
        
        
            
        #arguemnt and define variable
        server_gpu = 8
        node_size = world_size // server_gpu
        num_tp_groups = world_size // tensor_model_parallel_size
        num_inner_dp_size = server_gpu // tensor_model_parallel_size
        inner_num_tp_groups = server_gpu // tensor_model_parallel_size
        num_server_pp_group = node_size // pipeline_model_parallel_size
                 
        opt_pp_sequence_each = opt_pp_sequence_data[opt_ranking_num][0]
        
        # Build the tensor model-parallel groups. (Inner Server, it use nvlink network)
        # Build the tensor model-parallel groups.
          
        total_tp_group_ranks = []
        for i in range(num_tp_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            total_tp_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group
        print(f"total_tp_group_ranks : {total_tp_group_ranks}")   
        
        ##templory need data
        normal_pp_sequence = []
        for i in range(node_size):
            temp = []
            for j in range(inner_num_tp_groups):
                temp.append(list(total_tp_group_ranks[i*inner_num_tp_groups + j]))
            normal_pp_sequence.append(temp)
        print(f"normal_pp_sequence : {normal_pp_sequence}")        

        server_pp_group = []
        print(f"node_size : {node_size}")
        print(f"pipeline_model_parallel_size : {pipeline_model_parallel_size}")
        print(f"num_server_pp_group : {num_server_pp_group}")
        print(f"opt_pp_sequence_each : {opt_pp_sequence_each}")
        for i in range(num_server_pp_group):
            #server_pp_group.append(opt_pp_sequence_each[(i*pipeline_model_parallel_size):(i+1)*pipeline_model_parallel_size]) 
            server_pp_group.append(opt_pp_sequence_each[i])            
        print(f"length of server_pp_group {len(server_pp_group)}")
        print(f"server_pp_group : {server_pp_group}")   
        
        # Build the pipeline model-parallel groups. (Inter Server, it use ib network)

        total_pp_group_ranks = [] 
        

        for i in range(len(server_pp_group)):
            server_pp_group_each = server_pp_group[i]
            print(f"server_pp_group_each : {server_pp_group_each}")
            for k in range(inner_num_tp_groups):
                pp_group_ranks_each=[]
                for j in range(len(server_pp_group_each)):
                    pp_server = server_pp_group_each[j]
                    temp_rank = normal_pp_sequence[pp_server][k]
                    pp_group_ranks_each.append(temp_rank)
                pp_group_ranks_each = [list(x) for x in zip(*pp_group_ranks_each)]
                for p in range(len(pp_group_ranks_each)):   
                    ranks = pp_group_ranks_each[p]
                    total_pp_group_ranks.append(ranks)
                    group = torch.distributed.new_group(ranks)
                    if rank in ranks:
                        _PIPELINE_MODEL_PARALLEL_GROUP = group
                        _PIPELINE_GLOBAL_RANKS = ranks                    
                #print(f"pp_group_each : {pp_group_ranks_each}")   
        print(f"total_pp_group_ranks : {total_pp_group_ranks}") 
              
        # Build the embedding model-parallel groups. (Inter Server, it use ib network)             
        total_embedding_group_ranks = [] 
        for ranks in total_pp_group_ranks:
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
            else:
                embedding_ranks = ranks
            total_embedding_group_ranks.append(embedding_ranks)
            group = torch.distributed.new_group(embedding_ranks)
            if rank in embedding_ranks:
                _EMBEDDING_GROUP = group  
        print(f"total_embedding_group_ranks : {total_embedding_group_ranks}") 
        
        # Build the data-parallel groups. 
        all_data_parallel_group_ranks = []
        for i in range(pipeline_model_parallel_size):
            dp_group_ranks_each=[]
            for j in range(num_server_pp_group):
                server_pp_group_each = server_pp_group[j]     
                #print(f"server_pp_group_each : {server_pp_group_each}")  
                pp_server = server_pp_group_each[i] 
                for k in range(num_inner_dp_size):
                    temp_rank = normal_pp_sequence[pp_server][k]
                    dp_group_ranks_each.append(temp_rank)
            dp_group_ranks_each = [list(x) for x in zip(*dp_group_ranks_each)]
            
            for p in range(len(dp_group_ranks_each)):   
                ranks = dp_group_ranks_each[p]
                all_data_parallel_group_ranks.append(ranks)  
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _DATA_PARALLEL_GROUP = group   
        print(f"all_data_parallel_group_ranks : {all_data_parallel_group_ranks}")  
        
                    
        # Build the model-parallel groups.
        total_mp_group_ranks = []
        for i in range(data_parallel_size):
            ranks = [data_parallel_group_ranks[i]
                    for data_parallel_group_ranks in all_data_parallel_group_ranks]
            total_mp_group_ranks.append(ranks)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _MODEL_PARALLEL_GROUP = group  
        _MODEL_PARALLEL_GLOBAL_GROUP = total_mp_group_ranks
        print(f"total_mp_group_ranks : {total_mp_group_ranks}") 
                  
    
    else:   
        # Build the data-parallel groups.
        all_data_parallel_group_ranks = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(tensor_model_parallel_size):
                ranks = range(start_rank + j, end_rank,
                            tensor_model_parallel_size)
                all_data_parallel_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _DATA_PARALLEL_GROUP = group

        # Build the model-parallel groups.
        for i in range(data_parallel_size):
            ranks = [data_parallel_group_ranks[i]
                    for data_parallel_group_ranks in all_data_parallel_group_ranks]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _MODEL_PARALLEL_GROUP = group

        # Build the tensor model-parallel groups.
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size,
                        (i + 1) * tensor_model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group

        # Build the pipeline model-parallel groups and embedding groups
        # (first and last rank in each pipeline model-parallel group).
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size,
                        num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
            else:
                embedding_ranks = ranks
            group = torch.distributed.new_group(embedding_ranks)
            if rank in embedding_ranks:
                _EMBEDDING_GROUP = group
                
                
def get_world_last_rank():
    assert _MODEL_PARALLEL_GLOBAL_GROUP is not None, \
        "model parallel global group is not initialized"
    return _MODEL_PARALLEL_GLOBAL_GROUP[-1][-1]    


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if get_virtual_pipeline_model_parallel_world_size() is not None and \
            get_virtual_pipeline_model_parallel_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = \
            get_virtual_pipeline_model_parallel_world_size()
        if virtual_pipeline_model_parallel_world_size is not None and \
            get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_pipeline_model_parallel_first_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]



def get_pipeline_model_parallel_last_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]

def get_pipeline_model_parallel_next_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
