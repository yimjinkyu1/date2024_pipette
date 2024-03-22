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

from functools import reduce
import operator
import torch

from megatron import get_args
from megatron import mpu

import torch.distributed as dist 
import socket
import os
import json

# def _mapper_loader():
#     args = get_args()
    
#     map_dict = {}
#     if args.topo_aware:
#         orig_ranks = [i for i in range(args.world_size)]
#         for rank in orig_ranks:
#             if 16 <= rank and rank <= 23:
#                 map_dict[rank] = rank + 16
#             elif 24 <= rank and rank <= 31:
#                 map_dict[rank] = rank + 16
#             elif 32 <= rank and rank <= 39:
#                 map_dict[rank] = rank - 16
#             elif 40 <= rank and rank <= 47:
#                 map_dict[rank] = rank - 16
#             else:
#                 map_dict[rank] = rank
#         # todo - fill me
#         # need mapping table
#         # json file read 
#         # map_dict[rank] 
#         #pass
#     else:
#         orig_ranks = [i for i in range(args.world_size)]
#         for rank in orig_ranks:
             
#             map_dict[rank] = rank

#     return map_dict



def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                 use_ring_exchange=False, tensor_shape=None,
                 override_scatter_gather_tensors_in_pipeline=False,
                 dtype_=None, timers=None):
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        use_ring_exchange: boolean for whether torch.distributed.ring_exchange()
                           API should be used.
        tensor_shape: optional, use when the input sequence contains less
                      tokens than the default sequence length
        override_scatter_gather_tensors_in_pipeline: optional, this is used
                                                     when tensor_shape is
                                                     provided to overwide
                                                     scatter gather tensors
        dtype_: optional, this is used when tensor_shape is provied and what
                is the type of tensor_shape
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    
    if tensor_shape is None:
        tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
        
    jk_hostname = socket.gethostname()
    jk_dist_rank = dist.get_rank()
    jk_local_rank = 10000
    
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:

        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1) // \
            mpu.get_tensor_model_parallel_world_size()
    else:

        tensor_chunk_shape = tensor_shape
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float

    requires_grad = True
    if dtype_ is not None:

        dtype = dtype_
        requires_grad = False
        
    

    if recv_prev:
        tensor_recv_prev = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
    if recv_next:
        tensor_recv_next = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:
        if tensor_send_next is not None:
            tensor_send_next = mpu.split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = mpu.split_tensor_into_1d_equal_chunks(tensor_send_prev)


    # [Modified] topology aware or not
    #rank_mapper = _mapper_loader()
    #rank_mapper[mpu.get_pipeline_model_parallel_prev_rank()]
    
    p2p_timer_log=''
    # Send tensors in both the forward and backward directions as appropriate.
    if use_ring_exchange:
        torch.distributed.ring_exchange(tensor_send_prev=tensor_send_prev,
                                        tensor_recv_prev=tensor_recv_prev,
                                        tensor_send_next=tensor_send_next,
                                        tensor_recv_next=tensor_recv_next,
                                        group=mpu.get_pipeline_model_parallel_group())
    else:
        ops = []

        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_prev,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(send_prev_op)
            dest = mpu.get_pipeline_model_parallel_prev_rank()
            topo_dest = dest
            p2p_timer_log = "tensor_send_prev_from_to {}, {}".format(jk_hostname, dest)
            #print(f"hostname : {jk_hostname}, dist_rank : {jk_dist_rank},  send_prev_op, original dest {dest}, topo_dest {topo_dest}, p2p_communication.py 10")
            
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_prev,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(recv_prev_op)
            source = mpu.get_pipeline_model_parallel_prev_rank()
            topo_source = source
            p2p_timer_log = "{}: tensor_recv_prev_from_to {}, {}".format(p2p_timer_log, source, jk_hostname)
            #print(f"hostname : {jk_hostname}, dist_rank : {jk_dist_rank}, recv_prev_op, original source {source}, topo_source {topo_source}, p2p_communication.py 11") 
            
        if tensor_send_next is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_next,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(send_next_op)
            dest = mpu.get_pipeline_model_parallel_next_rank()
            topo_dest = dest
            p2p_timer_log = "{}: tensor_send_next_from_to {}, {}".format(p2p_timer_log, jk_hostname, dest)
            #print(f"hostname : {jk_hostname}, dist_rank : {jk_dist_rank}, send_next_op, original dest {dest}, topo_dest {topo_dest}, p2p_communication.py 12") 
            
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_next,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(recv_next_op)
            source = mpu.get_pipeline_model_parallel_next_rank()
            topo_source = source 
            p2p_timer_log = "{}: tensor_recv_next_from_to {}, {}".format(p2p_timer_log, source, jk_hostname)
            #print(f"hostname : {jk_hostname}, dist_rank : {jk_dist_rank},  recv_next_op, original source {source}, topo_source {topo_source}, p2p_communication.py 13") 
        
        timers(p2p_timer_log).start()        
        if len(ops) > 0:  
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        timers(p2p_timer_log).stop()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline:
        if recv_prev:
            tensor_recv_prev = mpu.gather_split_1d_tensor(
                tensor_recv_prev).view(tensor_shape).requires_grad_()

        if recv_next:
            tensor_recv_next = mpu.gather_split_1d_tensor(
                tensor_recv_next).view(tensor_shape).requires_grad_()

    return tensor_recv_prev, tensor_recv_next


def recv_forward(tensor_shape=None,
                 override_scatter_gather_tensors_in_pipeline=False,
                 dtype_=None, timers=None):
    """Receive tensor from previous rank in pipeline (forward receive)."""

    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('forward-recv').start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            override_scatter_gather_tensors_in_pipeline=\
                override_scatter_gather_tensors_in_pipeline,
            dtype_=dtype_, timers=timers)
        if timers is not None:
            timers('forward-recv').stop()
    return input_tensor


def recv_backward(timers=None):
    """Receive tensor from next rank in pipeline (backward receive)."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('backward-recv').start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True, timers=timers)
        if timers is not None:
            timers('backward-recv').stop()
    return output_tensor_grad


def send_forward(output_tensor, timers=None,
                 override_scatter_gather_tensors_in_pipeline=False,
                 dtype_=None):
    """Send tensor to next rank in pipeline (forward send)."""

    if not mpu.is_pipeline_last_stage():
        if timers is not None:
            timers('forward-send').start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            override_scatter_gather_tensors_in_pipeline=\
            override_scatter_gather_tensors_in_pipeline,
            dtype_=dtype_, timers=timers)
        if timers is not None:
            timers('forward-send').stop()


def send_backward(input_tensor_grad, timers=None):
    """Send tensor to previous rank in pipeline (backward send)."""
    if not mpu.is_pipeline_first_stage():
        if timers is not None:
            timers('backward-send').start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False, timers=timers)
        if timers is not None:
            timers('backward-send').stop()


def send_forward_recv_backward(output_tensor, timers=None):
    """Batched send and recv with next rank in pipeline."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('forward-send-backward-recv').start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True, timers=timers)
        if timers is not None:
            timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, timers=None):
    """Batched send and recv with previous rank in pipeline."""
    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('backward-send-forward-recv').start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False, timers=timers)
        if timers is not None:
            timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(output_tensor, recv_prev, timers=None):
    """Batched recv from previous rank and send to next rank in pipeline."""
    if timers is not None:
        timers('forward-send-forward-recv').start()
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False, timers=timers)
    if timers is not None:
        timers('forward-send-forward-recv').stop()
    return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next, timers=None):
    """Batched recv from next rank and send to previous rank in pipeline."""
    if timers is not None:
        timers('backward-send-backward-recv').start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next, timers=timers)
    if timers is not None:
        timers('backward-send-backward-recv').stop()
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
        output_tensor, input_tensor_grad, recv_prev,
        recv_next, timers=None):
    """Batched send and recv with previous and next ranks in pipeline."""
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').start()
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next, timers=timers)
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad
