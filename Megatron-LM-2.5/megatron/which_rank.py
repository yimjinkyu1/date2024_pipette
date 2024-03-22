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

"""General utilities."""


import torch
from megatron import get_args
from megatron import mpu


def is_last_rank_v2():
    args = get_args()
    
    if args.topo_aware:
        return torch.distributed.get_rank() == mpu.get_world_last_rank()
    else:
        return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)

def print_rank_last_v2(message):

    
    if torch.distributed.is_initialized():
        if is_last_rank_v2():
            #print(f"hostname : {jk_hostname}, dist_rank : {jk_dist_rank}, local_rank : {jk_local_rank}, message : {message}, print_rank_last 2, __init__.py")
            print("print_rank_last_v2 inner")
            print(message, flush=True)
    else:
        print(message, flush=True)