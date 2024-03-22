import pandas as pd 
import numpy as np 
import torch
import itertools 
import argparse

import time

import json

def __init__() : 
    return 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_file',type=str, default=None)
    parser.add_argument('--output_file',type=str, default=None)
    read_file = parser.parse_args().read_file
    output_file = parser.parse_args().output_file

    return read_file, output_file




def data_gpu_agg(data) : 
    data = data.groupby('Recv').sum() 
    data = data.T # Transpose  

    data = data.groupby(data.index).sum() # Calculate Sum  
    data = data.T # Transpose Again 

    # Set index/Column name 
    data.index.name = "Destination" # From 
    data.columns.name = "Source" # To

    # diagonal value 
    for i in range(len(data)) : 
        for j in range(len(data)) : 
            m = max(data.iloc[i])
            if i==j : 
                data.iloc[i][j] = m

    cols = data.columns
    data = np.array(data)
    data = data / (4*4) 
    return data, cols 


def calculate_pp_node(node_size, pp_node_sequence_list, send_bw_data, recv_bw_data):
    pp_node_sequence_sum = []  
    for pp in pp_node_sequence_list : 
        pp_latency_sum = 0
        for i in range(node_size-1):
            #print(pp[i]) 
            send_server = pp[i]
            recv_server = pp[i+1]
            send_bw = send_bw_data[send_server][recv_server]
            recv_bw = recv_bw_data[send_server][recv_server]
            avg_bw = (send_bw + recv_bw ) / 2
            send_latency = 1/avg_bw
            pp_latency_sum += send_latency
        pp_node_sequence_sum.append(pp_latency_sum)
        #for i in range(pipeline_model_parallel_size) : 
        #    locals()[f'node_{i}'] = pp[i] 
        #for i in range(pipeline_model_parallel_size-1) : 
        #    sum += data[eval('node_'+str(i))][eval('node_'+str(i+1))]
        #pp_node_sequence_sum.append(sum)

    # Calculate 
    pp_node_sequences = [(seq, sum) for seq, sum in zip(pp_node_sequence_list, pp_node_sequence_sum)]   # (permutaion, sum)
    print("not sosrt")
    print(pp_node_sequences)

    pp_node_sequences = sorted(pp_node_sequences, key=lambda x : x[1], reverse=False)  # sort by sum  # rank reverse 
    
    print("sort")
    print(pp_node_sequences)
    
    return pp_node_sequences 

def data_read(data, element_str):
    avg_temp = 0
    try:
        avg_temp = data[element_str]
        return avg_temp
    except:
        return 0 

def main(read_file, output_file) : 

    
    #print(f"pp_node_sequence_list: {pp_node_sequence_list}")
    
    with open(read_file, "r") as f:
        data = json.load(f)
    
    #MB/sec 
    iteration_time = data_read(data, "elapsed-time-per-iteration")
    log_iters = data_read(data,"log_iters")
    grad_norm = data_read(data,"grad-norm")
    forward_compute_time = data_read(data,"forward-compute_time")
    forward_recv_time = data_read(data,"forward-recv_time")
    forward_send_time = data_read(data,"forward-send_time")
    backward_compute_time = data_read(data,"backward-compute_time")
    backward_recv_time = data_read(data,"backward-recv_time")
    backward_send_time = data_read(data,"backward-send_time")
    backward_params_all_reduce_time = data_read(data,"backward-params-all-reduce_time")
    backward_embedding_all_reduce_time = data_read(data,"backward-embedding-all-reduce_time")
    
    # iteration_time = data["elapsed-time-per-iteration"]
    # log_iters = data["log_iters"]
    # grad_norm = data["grad-norm"]
    # forward_compute_time = data["forward-compute_time"]
    # forward_recv_time = data["forward-recv_time"]
    # forward_send_time = data["forward-send_time"]
    # backward_compute_time = data["backward-compute_time"]
    # backward_recv_time = data["backward-recv_time"]
    # backward_send_time = data["backward-send_time"]
    # backward_params_all_reduce_time = data["backward-params-all-reduce_time"]
    # backward_embedding_all_reduce_time = data["backward-embedding-all-reduce_time"]

    avg_grad_norm = np.average(grad_norm)
    avg_fwd_compute = np.average(forward_compute_time)
    avg_fwd_recv = np.average(forward_recv_time)
    avg_fwd_send = np.average(forward_send_time)
    avg_bwd_compute = np.average(backward_compute_time)
    avg_bwd_recv = np.average(backward_recv_time)
    avg_bwd_send = np.average(backward_send_time)
    avg_bwd_allreduce = np.average(backward_params_all_reduce_time)
    avg_emb_allreduce = np.average(backward_embedding_all_reduce_time)
    total_iteration_time = 0
    
    total_iteration_time = avg_grad_norm + avg_fwd_compute + avg_fwd_recv + avg_fwd_send + avg_bwd_compute + avg_bwd_recv + avg_bwd_send + avg_bwd_allreduce +  avg_emb_allreduce 
    
    print(f"elapsed_time_per_iteration : {iteration_time}")
    print(f"total_iteration_time : {total_iteration_time}")

    print(f"gard_norm : {avg_grad_norm}")
    print(f"forward_compute_time : {avg_fwd_compute}")
    print(f"forward_recv_time : {avg_fwd_recv}")
    print(f"forward_send_time : {avg_fwd_send}")
    print(f"backward_compute_time : {avg_bwd_compute}")
    print(f"backward_recv_time : {avg_bwd_recv}")
    print(f"backward_send_time : {avg_bwd_send}")
    print(f"backward_params_all_reduce_time : {avg_bwd_allreduce}")
    print(f"backward_embedding_all_reduce_time : {avg_emb_allreduce}")
    

    print(f"{total_iteration_time}")
    print(f"{avg_grad_norm}")
    print(f"{avg_fwd_compute}")
    print(f"{avg_fwd_recv}")
    print(f"{avg_fwd_send}")
    print(f"{avg_bwd_compute}")
    print(f"{avg_bwd_recv}")
    print(f"{avg_bwd_send}")
    print(f"{avg_bwd_allreduce}")
    print(f"{avg_emb_allreduce}")
    



    
    
    
    
    

    

if __name__ == '__main__': 
    read_file, output_file = get_arguments()
    main(read_file, output_file)
