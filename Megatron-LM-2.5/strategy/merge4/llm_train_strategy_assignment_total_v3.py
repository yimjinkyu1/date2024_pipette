import argparse
import joblib
from sklearn.neural_network import MLPRegressor
import csv
import json
import torch
import pandas as pd 
import os
import mh_simulated_annealing_opt as MHSimA #Simulated Annealing 
import mh_cp_solver as MHSat  #CP-SAT 
#import mh_genetic as MHGen  #Genetic 
#import mh_rf_learning as MHRf #Reinforcement Learning 

import random
import math
import time
import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument("--exp_id", type=int, default=1, help="put your experiment id")
parser.add_argument("--server", type=int, default=1, help="number of servers in use")
parser.add_argument("--model_type", type=str, default="GPT",  help='put your model type.')
parser.add_argument("--parameter_num", type=str, default="test",  help='put your model type.')
parser.add_argument('--global_batch_size', required=False, type=int, default=512, help='global batch size')
parser.add_argument('--micro_batch_size', required=False, type=int, default=8, help='micro batch size')
parser.add_argument("--ddp_impl", type=str, default="local",  help='put your model type.')
parser.add_argument('--fp_type', type=str, default="fp32", help='put your floating-point type.(fp32 or fp16 or df16)')
parser.add_argument('--train_iters', required=False, type=int, default=10, help='put iteration number')
parser.add_argument('--network_speed', required=False, type=str, default="200G", help="200G or 800G")
parser.add_argument('--q', default='admin_dev_gpu_all', type=str, required=False, help='lsf gpu queue.')
parser.add_argument('--previous_checkpoint', default=-1, type=int, required=False, help='checkpoint path')
parser.add_argument('--p2p_dummy_size', default=1, type=int, required=True, help='p2p_dummy_size.')
parser.add_argument('--num_layers', required=False, type=int, help='number of train model layer')
parser.add_argument('--hidden_size', required=False, type=int, help='hidden size of transformer model')
parser.add_argument('--num_attention_heads', required=False, type=int, help='attention node size of transformer model')
parser.add_argument('--seq_length', required=True, type=int, help='seq_length size of train')
parser.add_argument('--vocab_size', required=True, type=int, help='vocab size of train')
parser.add_argument('--write_file', required=True, type=str, help='put write csv file name')
parser.add_argument('--write_file_path', required=True, type=str, help='put write csv file path')
parser.add_argument('--pp_inter_bw', required=True, type=float, help='busbandwidth(GB/s) of sendrecv inter-node')
parser.add_argument('--dp_inter_bw', required=True, type=float, help='busbandwidth(GB/s) of allreduce inter-node')
parser.add_argument('--dp_inner_bw', required=True, type=float, help='busbandwidth(GB/s) of sendrecv inner-node')
parser.add_argument('--gpu_memory_check', required=False, type=bool, default=False, help='check gpu memory threshold')
parser.add_argument('--gpu_memory_size', required=False, type=float, default=80000, help='check gpu memory threshold')
parser.add_argument('--select_num', required=False, type=int, default=10, help='select recommend strategy')
parser.add_argument('--nccl_test_file', required=True, type=str, help='bandwidth profile file path')
parser.add_argument('--opt_output_file', required=True, type=str, help='optimization grouping/pipelining file path')
parser.add_argument('--opt_test_num', required=False, default=6, type=int, help='optimization grouping/pipelining test number')
parser.add_argument('--sa_initial_temp', type=int, default=1, help='initial temp for simulated annealing')
parser.add_argument('--sa_alpha', type=float, default=0.999, help='alpha for simulated annealing')
parser.add_argument('--sa_time_limit', type=int, default = 500, help='time limit of simulated annealing')
parser.add_argument('--nccl_algo', required=True, type=str, help='nccl algorithm Ring,Tree,CollNet')
parser.add_argument('--gpu_type', required=False, type=str, default='a100', help='gpu_type : v100, a100')
parser.add_argument('--fine_grained_algo', required=False, type=str, default='sa', help='fine_grained_algo : sa, sat, gen, rf')
parser.add_argument("--pred_model", type=str, default="MLP",  help='put your predict model type.')
parser.add_argument('--run_local', type=str, required=False, default="s2", help='submit scheducler job file(s2) or local server(local)')
parser.add_argument('--local_server_list', type=str, required=False, default="1102_32gpu", help='local server list')
parser.add_argument('--exp_alphabet', type=str, required=False, default="a", help='experiment alphabet(a,b,c,...')
parser.add_argument('--log_path', type=str, required=True, help='log path')
parser.add_argument('--b2_jobid', default=5555, type=int, help='scheduler job id (s2)')
parser.add_argument('--pp', default=16, type=int, help='pipeline')
parser.add_argument('--tp', default=8, type=int, help='pipeline')
args = parser.parse_args()

# MLP 모델 정의 (PyTorch 사용)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, 12)
        self.fc5 = nn.Linear(12, output_dim)
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, output_dim)

   
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
    
predict_model_path = "/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/merge4/predict_model/"

# if args.gpu_type == "a100" or args.gpu_type == "A100":
#     scaler = joblib.load(predict_model_path+'standard_scaler_model.pkl')
#     model_predict_gpu_mem = joblib.load(predict_model_path+'model_predict_gpu_mem.pkl')
#     model_predict_fwd_time = joblib.load(predict_model_path+'model_predict_fwd_time.pkl')
#     model_predict_bwd_time = joblib.load(predict_model_path+'model_predict_bwd_time.pkl')
#     args.gpu_memory_size = 70000
    
# if args.gpu_type == "a100" or args.gpu_type == "A100":
#     scaler = joblib.load(predict_model_path+'standard_scaler_model_real_result_a100.pkl')
#     model_predict_gpu_mem = joblib.load(predict_model_path+'model_predict_gpu_mem_real_result_a100_MLP_200_50000.pkl')
#     model_predict_fwd_time = joblib.load(predict_model_path+'model_predict_fwd_time_real_result_a100_MLP_200_50000.pkl')
#     model_predict_bwd_time = joblib.load(predict_model_path+'model_predict_bwd_time_real_result_a100_MLP_200_50000.pkl')
#     args.gpu_memory_size = 70000

# elif args.gpu_type == "v100" or args.gpu_type == "V100":
#     scaler = joblib.load(predict_model_path+'standard_scaler_model_real_result_v100.pkl')
#     model_predict_gpu_mem = joblib.load(predict_model_path+'model_predict_gpu_mem_real_result_v100_MLP_200_50000.pkl')
#     model_predict_fwd_time = joblib.load(predict_model_path+'model_predict_fwd_time_real_result_v100_MLP_200_50000.pkl')
#     model_predict_bwd_time = joblib.load(predict_model_path+'model_predict_bwd_time_real_result_v100_MLP_200_50000.pkl')
#     args.gpu_memory_size = 20000
# else:
#     print("Unkown gpu type")   
 
device = torch.device("cpu")

if args.gpu_type == "a100" or args.gpu_type == "A100":
    scaler = joblib.load(predict_model_path+'standard_scaler_model_real_result_a100_torch_origin_no_cpu.pkl')
    
    model_predict_gpu_mem = torch.load(predict_model_path+'model_predict_gpu_mem_a100_torch_origin_no_cpu.pkl',map_location=torch.device('cpu'))
    model_predict_fwd_time = torch.load(predict_model_path+'model_predict_fwd_time_a100_torch_origin_no_cpu.pkl',map_location=torch.device('cpu'))
    model_predict_bwd_time = torch.load(predict_model_path+'model_predict_bwd_time_a100_torch_origin_no_cpu.pkl',map_location=torch.device('cpu'))
    #model_predict_gpu_mem = model_predict_gpu_mem.to(device)
    #model_predict_fwd_time = model_predict_bwd_time.to(device)
    #model_predict_bwd_time = model_predict_bwd_time.to(device)
    args.gpu_memory_size = 70000

elif args.gpu_type == "v100" or args.gpu_type == "V100":
    scaler = joblib.load(predict_model_path+'standard_scaler_model_real_result_v100_torch_origin_no_cpu.pkl')
    model_predict_gpu_mem = torch.load(predict_model_path+'model_predict_gpu_mem_v100_torch_origin_no_cpu.pkl',map_location=torch.device('cpu'))
    model_predict_fwd_time = torch.load(predict_model_path+'model_predict_fwd_time_v100_torch_origin_no_cpu.pkl',map_location=torch.device('cpu'))
    model_predict_bwd_time = torch.load(predict_model_path+'model_predict_bwd_time_v100_torch_origin_no_cpu.pkl',map_location=torch.device('cpu'))
    #model_predict_gpu_mem = model_predict_gpu_mem.to(device)
    #model_predict_fwd_time = model_predict_bwd_time.to(device)
    #model_predict_bwd_time = model_predict_bwd_time.to(device)
    args.gpu_memory_size = 20000
else:
    print("Unkown gpu type")   
    
    
# elif args.gpu_type == "v100" or args.gpu_type == "V100":
# scaler = joblib.load(predict_model_path+'standard_scaler_model_v100.pkl')
# model_predict_gpu_mem = joblib.load(predict_model_path+'model_predict_gpu_mem_v100.pkl')
# model_predict_fwd_time = joblib.load(predict_model_path+'model_predict_fwd_time_v100.pkl')
# model_predict_bwd_time = joblib.load(predict_model_path+'model_predict_bwd_time_v100.pkl')
# args.gpu_memory_size = 25000

# GPT-8.3B
global_layers = args.num_layers
global_hidden_size = args.hidden_size
global_num_attention_heads = args.num_attention_heads
global_seq_length = args.seq_length

servers = args.server
global_bs = args.global_batch_size

gpu_in_server = 8
gpus = servers * gpu_in_server
tp_range = [1, 2, 4, 8]
fp_type = args.fp_type  

pp_inter_bw = args.pp_inter_bw

dp_inter_bw = args.dp_inter_bw
dp_inner_bw = args.dp_inner_bw 



def find_divisors(n):
    divisors = []
    micro_range = [1, 2, 4, 8]
    #for i in range(1, n + 1):
    for i in micro_range:
        if n % i == 0:
            divisors.append(i)
            #if i % 8 == 0:
            #    divisors.append(i)
    return divisors

def find_divisors_v2(n):
    divisors = []
    micro_range = args.micro_batch_size
    if n % micro_range == 0:
        divisors.append(micro_range)
        #if i % 8 == 0:
        #    divisors.append(i)
    return divisors



def get_pp_msg_size(micro_batch_size):
    
    seq_length = global_seq_length
    hidden_size = global_hidden_size
    fp_type = args.fp_type
    
    # 1) calculate pipeline parallel message size
    pp_msg_size = micro_batch_size * seq_length * hidden_size

    if fp_type == "fp16" or fp_type == "bf16":
        pp_msg_size = pp_msg_size * 2
    else:
        pp_msg_size = pp_msg_size * 4
        
    pp_msg_size_gigabyte = pp_msg_size / 10**9
    
    return pp_msg_size_gigabyte


def get_dp_msg_size(in_layers, tp):
    
    vocab_size = args.vocab_size #vocab_size = 50256
    model_type = args.model_type
    hidden_size = global_hidden_size
    fp_type = args.fp_type
    
    # 2) calculate dp_cost for the first data parallel group
    if (model_type == "BERT"):
        param_weight = 12
    elif (model_type == "GPT"):
        param_weight = 16
    else:
        print('Unknown model or Unknown parameter number')
        exit()
        
    emb_param_count = hidden_size * vocab_size
    one_transformer_param_count = param_weight * hidden_size ** 2
    total_transformer_param_count = in_layers * one_transformer_param_count
    param_count = ( emb_param_count + total_transformer_param_count ) / tp
    
    if fp_type == "fp16" or fp_type == "bf16":
        dp_msg_size = param_count * 2
    else:
        dp_msg_size = param_count * 4
    
    dp_msg_size_gigabyte = dp_msg_size / 10**9 
    
    return dp_msg_size_gigabyte
    
# pipeline communication time 
def get_cost_pp_amp(micro_batch_size, pp):

    pp_msg_size_gigabyte = get_pp_msg_size(micro_batch_size)

    # activation/gradient message sendrecv time (sec) 
    cost_inter_pp = 2*(pp-1)*(pp_msg_size_gigabyte / pp_inter_bw)  # seconds
    cost_inter_pp = cost_inter_pp * 10**3 #transfrom to milliseconds

      
    return cost_inter_pp

# data parallel communication time 
def get_cost_dp_amp(tp, dp, in_layer):
    
    dp_msg_size_gigabyte = get_dp_msg_size(in_layer, tp)
    
    min_dp_bw = min(dp_inner_bw, dp_inter_bw)
    
    cost_dp =  ( 2 * (dp - 1) * dp_msg_size_gigabyte )/ (dp * min_dp_bw)  
    
    cost_dp = cost_dp * 10**3
    
    return cost_dp


# data parallel communication time 
def get_cost_dp_split(tp, dp, inner_dp, in_layer):
    
    dp_msg_size_gigabyte = get_dp_msg_size(in_layer, tp)
    
    # cost inner data parallel 
    inter_dp = dp // inner_dp
    cost_inner_dp = 0
    if inner_dp != 1:
        cost_inner_dp = ( 2 * (inner_dp - 1) * dp_msg_size_gigabyte )/ (inner_dp * dp_inner_bw)  #seconds 
        cost_inner_dp = cost_inner_dp * 10**3   
        cost_inner_dp = 2*cost_inner_dp 
    
    # cost inter data parallel 
    cost_inter_dp = 0
    if inter_dp != 1: 
        cost_inter_dp = ( 2 * (inter_dp - 1)  * dp_msg_size_gigabyte ) / (inter_dp * dp_inter_bw)
        cost_inter_dp = cost_inter_dp * 10**3 
    
    return cost_inner_dp, cost_inter_dp

#After Date [JK]
def get_cost_opt(tp, pp, dp,inner_dp, in_layer, micro_batch_size, output_file, fine_algo):
    
    inter_dp = dp // inner_dp 
    
    pp_msg_size_gigabyte = get_pp_msg_size(micro_batch_size)
    dp_msg_size_gigabyte = get_dp_msg_size(in_layer, tp)
    
    pp_group = servers // pp # 16//8 = 2
    dp_group = pp # 8
    
    #1) Simulated Annealing 
    if fine_algo == 'sa':
        MetaHeuristic = MHSimA.MetaHeuristics(read_file=args.nccl_test_file, output_file=output_file, node_size=servers, pp_group=pp_group, dp_group=dp_group, inter_dp_size=inter_dp, inner_dp_size=inner_dp, pp_msg_size=pp_msg_size_gigabyte, dp_msg_size=dp_msg_size_gigabyte, initial_temp=args.sa_initial_temp, alpha=args.sa_alpha, time_limit=args.sa_time_limit)
    elif fine_algo == 'sat':
        MetaHeuristic = MHSimA.MetaHeuristics(read_file=args.nccl_test_file, output_file=output_file, node_size=servers, pp_group=pp_group, dp_group=dp_group, inter_dp_size=inter_dp, inner_dp_size=inner_dp, pp_msg_size=pp_msg_size_gigabyte, dp_msg_size=dp_msg_size_gigabyte, initial_temp=args.sa_initial_temp, alpha=args.sa_alpha, time_limit=args.sa_time_limit)
    elif fine_algo == 'gen':
        MetaHeuristic = MHSimA.MetaHeuristics(read_file=args.nccl_test_file, output_file=output_file, node_size=servers, pp_group=pp_group, dp_group=dp_group, inter_dp_size=inter_dp, inner_dp_size=inner_dp, pp_msg_size=pp_msg_size_gigabyte, dp_msg_size=dp_msg_size_gigabyte, initial_temp=args.sa_initial_temp, alpha=args.sa_alpha, time_limit=args.sa_time_limit)
    elif fine_algo == 'rf':
        MetaHeuristic = MHSimA.MetaHeuristics(read_file=args.nccl_test_file, output_file=output_file, node_size=servers, pp_group=pp_group, dp_group=dp_group, inter_dp_size=inter_dp, inner_dp_size=inner_dp, pp_msg_size=pp_msg_size_gigabyte, dp_msg_size=dp_msg_size_gigabyte, initial_temp=args.sa_initial_temp, alpha=args.sa_alpha, time_limit=args.sa_time_limit)
    else:
        MetaHeuristic = MHSimA.MetaHeuristics(read_file=args.nccl_test_file, output_file=output_file, node_size=servers, pp_group=pp_group, dp_group=dp_group, inter_dp_size=inter_dp, inner_dp_size=inner_dp, pp_msg_size=pp_msg_size_gigabyte, dp_msg_size=dp_msg_size_gigabyte, initial_temp=args.sa_initial_temp, alpha=args.sa_alpha, time_limit=args.sa_time_limit)
    #2) CP-SAT
    #MetaHeuristic = MHSimA.MetaHeuristics(read_file=args.nccl_test_file, output_file=output_file, node_size=servers, pp_group=pp_group, dp_group=dp_group, inter_dp_size=inter_dp, inner_dp_size=inner_dp, pp_msg_size=pp_msg_size_gigabyte, dp_msg_size=dp_msg_size_gigabyte, initial_temp=args.sa_initial_temp, alpha=args.sa_alpha, time_limit=args.sa_time_limit)
    
    #3) Genetic 
    #MetaHeuristic = MHSimA.MetaHeuristics(read_file=args.nccl_test_file, output_file=output_file, node_size=servers, pp_group=pp_group, dp_group=dp_group, inter_dp_size=inter_dp, inner_dp_size=inner_dp, pp_msg_size=pp_msg_size_gigabyte, dp_msg_size=dp_msg_size_gigabyte, initial_temp=args.sa_initial_temp, alpha=args.sa_alpha, time_limit=args.sa_time_limit)
    
    #4) Reinforement Learning 
    #MetaHeuristic = MHSimA.MetaHeuristics(read_file=args.nccl_test_file, output_file=output_file, node_size=servers, pp_group=pp_group, dp_group=dp_group, inter_dp_size=inter_dp, inner_dp_size=inner_dp, pp_msg_size=pp_msg_size_gigabyte, dp_msg_size=dp_msg_size_gigabyte, initial_temp=args.sa_initial_temp, alpha=args.sa_alpha, time_limit=args.sa_time_limit)
    
    MetaHeuristic.n_top_low = int(args.opt_test_num/2)

    
    cost_inter_pp, cost_inter_dp, cost_inner_dp, top_low = MetaHeuristic.sa_run()
    
    
    return cost_inter_pp, cost_inter_dp, cost_inner_dp, top_low

def dict_insert_item(dict, key, value):
    

    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)
        
    return dict 



#df = pd.DataFrame(columns=['exp_id','sub_test_id','gpus','model_type','parameter_num','global_batch_size','micro_batch_size','ddp_impl','fp_type','tp','pp','dp','train_iters','network_speed','q','previous_checkpoint','topo_aware','pipeline_opt_algo','pipeline_ranking','p2p_dummy_size','num_layers','hidden_size','num_attention_heads','predict','gpu_mem','fwd_time','bwd_time','cost_total','cost_pp','cost_dp','cost_rest', 'cost_inter_pp','cost_execution','cost_inner_dp','cost_inter_dp'])

df = pd.DataFrame(columns=['exp_id','sub_test_id','gpus','model_type','parameter_num','global_batch_size','micro_batch_size','ddp_impl','fp_type','tp','pp','dp','train_iters','network_speed','q','previous_checkpoint','p2p_dummy_size','num_layers','hidden_size','num_attention_heads','nccl_algo','run_local','local_server_list','predict','gpu_mem','fwd_time','bwd_time','cost_total','cost_pp','cost_dp','cost_rest','cost_inter_pp','cost_execution','cost_inner_dp','cost_inter_dp','cost_total_iterl','cost_pp_iterl','cost_dp_iterl','cost_rest_iterl','cost_inter_pp_iterl','cost_inner_dp_iterl','cost_inter_dp_iterl','cost_total_split','cost_pp_split','cost_dp_split','cost_rest_split','cost_inter_pp_split','cost_inner_dp_split','cost_inter_dp_split','cost_total_opt','cost_pp_opt','cost_dp_opt','cost_rest_opt','cost_inter_pp_opt','cost_inner_dp_opt','cost_inter_dp_opt','cost_total_iterl_split','cost_pp_iterl_split','cost_dp_iterl_split','cost_rest_iterl_split', 'cost_inter_pp_iterl_split','cost_inner_dp_iterl_split','cost_inter_dp_iterl_split', 'cost_total_iterl_split_opt','cost_pp_iterl_split_opt','cost_dp_iterl_split_opt','cost_rest_iterl_split_opt','cost_inter_pp_iterl_split_opt','cost_inner_dp_iterl_split_opt','cost_inter_dp_iterl_split_opt','log_path','gpu_type','b2_jobid','exp_alphabet','opt_output_file','write_file_str','fine_grained_algo'])

pd.options.display.max_columns = None

#df = pd.DataFrame(columns=['exp_id','sub_test_id','gpus','model_type','parameter_num','global_batch_size','micro_batch_size','ddp_impl','fp_type','tp','pp','dp','train_iters','network_speed','q','previous_checkpoint','topo_aware','pipeline_opt_algo','pipeline_ranking','p2p_dummy_size','num_layers','hidden_size','num_attention_heads','nccl_algo','run_local','local_server_list','predict','gpu_mem','fwd_time','bwd_time','cost_total','cost_pp','cost_dp','cost_rest', 'cost_inter_pp','cost_execution','cost_inner_dp','cost_inter_dp','cost_total_iterl','cost_pp_iterl','cost_dp_iterl','cost_rest_iterl', 'cost_inter_pp_iterl','cost_inner_dp_iterl','cost_inter_dp_iterl','cost_total_split','cost_pp_split','cost_dp_split','cost_rest_split', 'cost_inter_pp_split','cost_inner_dp_split','cost_inter_dp_split','cost_total_opt','cost_pp_opt','cost_dp_opt','cost_rest_opt', 'cost_inter_pp_opt','cost_inner_dp_opt','cost_inter_dp_opt','log_path','gpu_type','b2_jobid','exp_alphabet','opt_output_file'])

#df= pd.DataFrame()


print("start strategy")
sub_test_id = 0
mh_pipette_time_total = 0.0 
model_pipette_time_total = 0.0 
start_time = datetime.now()
#for pp in range(min(servers, global_layers), 0, -1):
for pp in [args.pp]:
    if servers % pp == 0 and global_layers % pp == 0:
        in_train_layer = global_layers // pp
        #print(f"train layer : {train_layer}")
        #for tp in tp_range:
        for tp in [args.tp]:
            dp = (gpus // pp) // tp
            if isinstance(dp, int):
                mini_bs = global_bs // dp
                if isinstance(mini_bs, int):
                    for micro_bs in find_divisors_v2(mini_bs):
                        iteration_time = 0
                    
                        # gas (number of micro-batches in a pipeline)
                        gas = mini_bs // micro_bs
                        inner_dp = gpu_in_server // tp
                        interleave_group_num = gas // pp 
                        
                    
                        #print(f"pp : {pp}, tp : {tp}, dp : {dp}, inner_dp : {inner_dp}, global_bs: {global_bs}, mini_bs : {mini_bs}, micro_bs : {micro_bs}, \
#global_layers : {global_layers}, layers: {in_train_layer}, hidden_size : {global_hidden_size}, \
#attention_heads : {global_num_attention_heads}")
                        
                        #print(f"hidden_size: {global_hidden_size}, attention_heads: {global_num_attention_heads}, tp: {tp}, inner_dp: {inner_dp}, lyaers: {in_train_layer}, micro_bs: {micro_bs}, mini_bs: {mini_bs}")
                        
                        print(f"world_size: {gpus}, global_num_layers: {global_layers}, hidden_size: {global_hidden_size}, attention_heads: {global_num_attention_heads}, tp: {tp}, pp: {pp}, dp: {dp}, in_lyaers: {in_train_layer}, micro_bs: {micro_bs},global_bs: {global_bs},  mini_bs: {mini_bs}")

                        #model_input_data = [[global_hidden_size, global_num_attention_heads, tp, inner_dp, in_train_layer, micro_bs, mini_bs]]
                        model_input_data = [[gpus, global_layers, global_hidden_size, global_num_attention_heads, tp, pp, dp, in_train_layer, micro_bs, global_bs, mini_bs]]
                        model_input_data_transform = scaler.transform(model_input_data)
                        
                        #print(model_input_data_transform)
                        #pred_gpu_mem_max = model_predict_gpu_mem.predict(model_input_data_transform)
                        #pred_fwd_time = model_predict_fwd_time.predict(model_input_data_transform)
                        #pred_bwd_time = model_predict_bwd_time.predict(model_input_data_transform)
                        model_input_data_transform = torch.FloatTensor(model_input_data_transform)
                        model_input_data_transform = model_input_data_transform.to(device)
                        
                        model_start_time = datetime.now()
                        with torch.no_grad():
                            model_predict_gpu_mem.eval()
                            model_predict_fwd_time.eval()
                            model_predict_bwd_time.eval()
                            
                            pred_gpu_mem_max = model_predict_gpu_mem(model_input_data_transform)
                            pred_fwd_time = model_predict_fwd_time(model_input_data_transform)
                            pred_bwd_time = model_predict_bwd_time(model_input_data_transform)
                            
                            pred_gpu_mem_max = pred_gpu_mem_max.numpy()
                            pred_fwd_time = pred_fwd_time.numpy()
                            pred_bwd_time = pred_bwd_time.numpy()
                            
                        pred_gpu_mem_max = pred_gpu_mem_max[0][0]
                        pred_fwd_time = pred_fwd_time[0][0]
                        pred_bwd_time = pred_bwd_time[0][0]
                        
                        model_end_time = datetime.now()
                        model_pipette_time =  model_end_time - model_start_time
                        model_pipette_time_milli = model_pipette_time.total_seconds()
                        model_pipette_time_total += model_pipette_time_milli
                        #pred_micro_bs_fwd_time = pred_fwd_time / gas
                        #pred_micro_bs_bwd_time = pred_bwd_time / gas 
                        
                        pred_micro_bs_fwd_time = pred_fwd_time 
                        pred_micro_bs_bwd_time = pred_bwd_time 

                        cost_execution = pp*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time) #milliseconds
                        
                        #1)amp
                        cost_inner_dp = 0 
                        #cost_rest = (gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time) #milliseconds
                        cost_rest = (gas-1)*(3*pred_micro_bs_fwd_time) #milliseconds
                        cost_inter_pp = get_cost_pp_amp(micro_bs, pp)
                        cost_execution_amp = pp*(3*pred_micro_bs_fwd_time) #milliseconds
                        cost_one_pp = cost_inter_pp + cost_execution_amp
                        cost_many_pp = cost_one_pp
                        cost_pp = cost_rest + cost_many_pp
                        cost_inter_dp = get_cost_dp_amp(tp, dp, in_train_layer)
                        cost_dp = cost_inner_dp + cost_inter_dp
                        cost_total = cost_pp + cost_dp 
                        
                        #2)amp+interleave (_iterl) = hidden_path (latency_model)
                        cost_inner_dp_iterl = 0 
                        cost_inter_pp_iterl = get_cost_pp_amp(micro_bs, pp)
                        cost_one_pp_iterl = cost_inter_pp_iterl + cost_execution 
                        if interleave_group_num >= 1:
                            cost_many_pp_iterl = interleave_group_num * cost_one_pp_iterl    
                            interleave_gas = pp
                            cost_rest_iterl =(interleave_gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time)
                        else:
                            cost_many_pp_iterl = cost_one_pp_iterl
                            cost_rest_iterl =(gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time) #straggler time (milliseconds)
                        cost_pp_iterl = cost_rest_iterl + cost_many_pp_iterl
                        cost_inter_dp_iterl = get_cost_dp_amp(tp, dp, in_train_layer)
                        cost_dp_iterl = cost_inner_dp_iterl + cost_inter_dp_iterl
                        cost_total_iterl = cost_pp_iterl + cost_dp_iterl 
                        
                        #3)amp+splitdp (_split) = inner_dp + inter_dp => get_cost_dp_split 
                        cost_inner_dp_split = 0
                        cost_rest_split = (gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time) #straggler time (milliseconds)
                        cost_inter_pp_split = get_cost_pp_amp(micro_bs, pp)
                        cost_one_pp_split = cost_inter_pp_split + cost_execution
                        cost_many_pp_split = cost_one_pp_split
                        cost_pp_split = cost_rest_split + cost_many_pp_split
                        cost_inner_dp_split, cost_inter_dp_split = get_cost_dp_split(tp, dp, inner_dp, in_train_layer)
                        cost_dp_split = cost_inner_dp_split + cost_inter_dp_split
                        cost_total_split = cost_pp_split + cost_dp_split 
                        
                        
                        #4)amp+opt(_opt = splitdp+real network) fine_grained => get_cost_opt 
                        #After Date [JK]
                        opt_output_file = args.opt_output_file + '_' +str(args.exp_id) + '_' + str(sub_test_id) + '_' +args.fine_grained_algo
                        cost_inner_dp_opt = 0 
                        cost_rest_opt = (gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time) #milliseconds
                        
                        mh_start_time = datetime.now()
                        
                        #After Date [JK]
                        cost_inter_pp_opt, cost_inter_dp_opt, cost_inner_dp_opt, top_low_opt = get_cost_opt(tp, pp, dp,inner_dp, in_layer=in_train_layer, micro_batch_size=micro_bs, output_file = opt_output_file, fine_algo=args.fine_grained_algo)
                        
                        
                        mh_end_time = datetime.now()
                        mh_pipette_time = mh_end_time - mh_start_time
                        mh_pipette_time_milli = mh_pipette_time.total_seconds()
                        mh_pipette_time_total += mh_pipette_time_milli
                        
                        cost_one_pp_opt = cost_inter_pp_opt + cost_execution
                        cost_many_pp_opt = cost_one_pp_opt
                        cost_pp_opt = cost_rest_opt + cost_many_pp_opt
                        cost_dp_opt = cost_inner_dp_opt + cost_inter_dp_opt
                        cost_total_opt = cost_pp_opt + cost_dp_opt
                        
                        #5)amp+interleave+splitdp (_iterl_split) 2) + 3) so no fine_grained 
                        cost_inner_dp_iterl_split = 0 
                        cost_inter_pp_iterl_split = get_cost_pp_amp(micro_bs, pp)
                        cost_one_pp_iterl_split = cost_inter_pp_iterl_split + cost_execution 
                        if interleave_group_num >= 1:
                            cost_many_pp_iterl_split = interleave_group_num * cost_one_pp_iterl_split    
                            interleave_gas = pp
                            cost_rest_iterl_split =(interleave_gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time)
                        else:
                            cost_many_pp_iterl_split = cost_one_pp_iterl_split
                            cost_rest_iterl_split =(gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time)
                        cost_pp_iterl_split = cost_rest_iterl_split + cost_many_pp_iterl_split
                        cost_inner_dp_iterl_split, cost_inter_dp_iterl_split = get_cost_dp_split(tp, dp, inner_dp, in_train_layer)
                        cost_dp_iterl_split = cost_inner_dp_iterl_split + cost_inter_dp_iterl_split
                        cost_total_iterl_split = cost_pp_iterl_split + cost_dp_iterl_split 
                        
                        
                        #6)our(amp+interleave+splitdp+opt) (_iterl_split_opt) 2) + 4) so latency_model + fine+grained  
                        cost_inner_dp_iterl_split_opt = 0 
                        cost_inter_pp_iterl_split_opt = cost_inter_pp_opt
                        #cost_inter_pp_iterl_split_opt = get_cost_pp_amp(micro_bs, pp)
                        cost_one_pp_iterl_split_opt = cost_inter_pp_opt + cost_execution 
                        # cost_one_pp_iterl_split_opt = cost_inter_pp_iterl_split_opt + cost_execution
                        if interleave_group_num >= 1:
                            cost_many_pp_iterl_split_opt = interleave_group_num * cost_one_pp_iterl_split_opt    
                            interleave_gas = pp
                            cost_rest_iterl_split_opt =(interleave_gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time)
                        else:
                            cost_many_pp_iterl_split_opt = cost_one_pp_iterl_split_opt
                            cost_rest_iterl_split_opt =(gas-1)*(pred_micro_bs_fwd_time + pred_micro_bs_bwd_time)
                        cost_pp_iterl_split_opt = cost_rest_iterl_split_opt + cost_many_pp_iterl_split_opt
                        cost_inner_dp_iterl_split_opt = cost_inner_dp_opt
                        cost_inter_dp_iterl_split_opt = cost_inter_dp_opt
                        #cost_inner_dp_iterl_split_opt, cost_inter_dp_iterl_split_opt = get_cost_dp_split(tp, dp, inner_dp, in_train_layer)
                        cost_dp_iterl_split_opt = cost_inner_dp_iterl_split_opt + cost_inter_dp_iterl_split_opt
                        cost_total_iterl_split_opt = cost_pp_iterl_split_opt + cost_dp_iterl_split_opt  
                
                        

                        print(f"gpu_mem : {pred_gpu_mem_max}, fwd_time : {pred_fwd_time}, bwd_time : {pred_bwd_time}")
                        print("==================================================================================")

                        new_data = {'exp_id':args.exp_id,'sub_test_id':sub_test_id, 'gpus':gpus,'model_type':args.model_type, 'parameter_num':args.parameter_num,'global_batch_size':global_bs,'micro_batch_size':micro_bs,'ddp_impl':args.ddp_impl,'fp_type':args.fp_type, 'tp':tp,'pp':pp,'dp':dp,'train_iters':args.train_iters,'network_speed':args.network_speed,'q':args.q, 'previous_checkpoint':args.previous_checkpoint,'p2p_dummy_size':args.p2p_dummy_size,'num_layers':global_layers,'hidden_size':global_hidden_size,'num_attention_heads':global_num_attention_heads,'nccl_algo':args.nccl_algo,'run_local':args.run_local,'local_server_list':args.local_server_list,'predict':'predict','gpu_mem':pred_gpu_mem_max,'fwd_time':pred_fwd_time,'bwd_time':pred_bwd_time,'cost_total':cost_total,'cost_pp':cost_pp,'cost_dp':cost_dp,'cost_rest':cost_rest, 'cost_inter_pp':cost_inter_pp,'cost_execution':cost_execution,'cost_inner_dp':cost_inner_dp,'cost_inter_dp':cost_inter_dp,'cost_total_iterl':cost_total_iterl,'cost_pp_iterl':cost_pp_iterl,'cost_dp_iterl':cost_dp_iterl,'cost_rest_iterl':cost_rest_iterl, 'cost_inter_pp_iterl':cost_inter_pp_iterl,'cost_inner_dp_iterl':cost_inner_dp_iterl,'cost_inter_dp_iterl':cost_inter_dp_iterl,'cost_total_split':cost_total_split,'cost_pp_split':cost_pp_split,'cost_dp_split':cost_dp_split,'cost_rest_split':cost_rest_split, 'cost_inter_pp_split':cost_inter_pp_split,'cost_inner_dp_split':cost_inner_dp_split,'cost_inter_dp_split':cost_inter_dp_split,'cost_total_opt':cost_total_opt,'cost_pp_opt':cost_pp_opt,'cost_dp_opt':cost_dp_opt,'cost_rest_opt':cost_rest_opt, 'cost_inter_pp_opt':cost_inter_pp_opt,'cost_inner_dp_opt':cost_inner_dp_opt,'cost_inter_dp_opt':cost_inter_dp_opt,'cost_total_iterl_split':cost_total_iterl_split,'cost_pp_iterl_split':cost_pp_iterl_split,'cost_dp_iterl_split':cost_dp_iterl_split,'cost_rest_iterl_split':cost_rest_iterl_split,'cost_inter_pp_iterl_split':cost_inter_pp_iterl_split,'cost_inner_dp_iterl_split':cost_inner_dp_iterl_split,'cost_inter_dp_iterl_split':cost_inter_dp_iterl_split, 'cost_total_iterl_split_opt':cost_total_iterl_split_opt,'cost_pp_iterl_split_opt':cost_pp_iterl_split_opt,'cost_dp_iterl_split_opt':cost_dp_iterl_split_opt,'cost_rest_iterl_split_opt':cost_rest_iterl_split_opt, 'cost_inter_pp_iterl_split_opt':cost_inter_pp_iterl_split_opt,'cost_inner_dp_iterl_split_opt':cost_inner_dp_iterl_split_opt,'cost_inter_dp_iterl_split_opt':cost_inter_dp_iterl_split_opt,'log_path':args.log_path,'gpu_type':args.gpu_type,'b2_jobid':args.b2_jobid,'exp_alphabet':args.exp_alphabet,'opt_output_file':opt_output_file,'write_file_str':args.write_file,'fine_grained_algo':args.fine_grained_algo}
                        print(new_data)
                        df = df.append(new_data, ignore_index=True)
                        print(df)
                        

                        sub_test_id += 1
end_time = datetime.now()
pipette_time = end_time - start_time
pipette_time_milli = pipette_time.total_seconds() 
print(f"pipette optimization time : {pipette_time_milli}")
print(f"pipette mh_pipette_time_total : {mh_pipette_time_total}")
print(f"model_pipette_time_total : {model_pipette_time_total}")


df = df.astype({'exp_id':'int','sub_test_id':'int','gpus':'int','global_batch_size':'int','micro_batch_size':'int','previous_checkpoint':'int','tp':'int','pp':'int','dp':'int','train_iters':'int','p2p_dummy_size':'int','num_layers':'int','hidden_size':'int','num_attention_heads':'int'})

pd.options.display.float_format = '{:.4f}'.format
sort_df = df.sort_values(by='cost_total')

select_test_num = args.select_num 
amp_sort_df = df.sort_values('cost_total')
head_amp_sort_df = amp_sort_df.head(select_test_num)

opt_head_amp_df = pd.DataFrame()

for index, row in head_amp_sort_df.iterrows():
    row_data = pd.DataFrame(row)
    row_data = row_data.transpose()
    row_data['opt_ranking_num'] = [-2]
    row_data['topo_aware'] = ['False']
    opt_head_amp_df = opt_head_amp_df.append(row_data,ignore_index=True)

if args.gpu_memory_check == True:
    condition = df['gpu_mem'] < args.gpu_memory_size
    check_gpu_mem_df = df[condition]
    sort_check_gpu_mem_df = check_gpu_mem_df.sort_values(by='cost_total_iterl_split_opt')
    
    over_condition = df['gpu_mem'] > args.gpu_memory_size 
    over_gpu_mem_df = df[over_condition]
    over_gpu_mem_df = over_gpu_mem_df.sort_values(by='gpu_mem')
    
else:
    check_gpu_mem_df = df
    sort_check_gpu_mem_df = sort_df

#select_test_num = args.select_num*(args.opt_test_num+1)    
head_sort_check_gpu_mem_df = sort_check_gpu_mem_df.head(select_test_num)
tail_sort_check_gpu_mem_df = sort_check_gpu_mem_df.tail(select_test_num)

mid_start = len(sort_check_gpu_mem_df) // 2
mid_end = mid_start + select_test_num
mid_sort_check_gpu_mem_df = sort_check_gpu_mem_df[mid_start:mid_end]

#recommend_df = pd.concat([head_sort_check_gpu_mem_df, tail_sort_check_gpu_mem_df])
recommend_df = head_sort_check_gpu_mem_df
recommend_df = recommend_df.drop_duplicates(['sub_test_id'])

opt_recommend_df = pd.DataFrame()
recommend_row_size = recommend_df.shape[0]

print(recommend_df.shape[0])
#for i in range(0,recommend_row_size,1):
    #row_data = pd.DataFrame(recommend_df.loc[i,:])

for index, row in recommend_df.iterrows():
    row_data = pd.DataFrame(row)
    row_data = row_data.transpose()
    row_data['opt_ranking_num'] = [-1]
    row_data['topo_aware'] = ['False']
    opt_recommend_df = opt_recommend_df.append(row_data,ignore_index=True)
    for opt_ranking_num in range(0,args.opt_test_num, 1):
        row_data['opt_ranking_num'] = [opt_ranking_num]
        row_data['topo_aware'] = ['True']
        opt_recommend_df = opt_recommend_df.append(row_data,ignore_index=True)

# opt_recommend_df = pd.DataFrame()
# for row in recommend_df.itertuples():
#     for opt_ranking_num in range(0,args.opt_test_num, 1):
#         new_data = pd.DataFrame({'opt_ranking_num':opt_ranking_num, 'topo_aware':True})   
#         opt_recommend_df = opt_recommend_df.append(pd.concat([row, new_data]),ignore_index=True)  
#     new_data = pd.DataFrame({'opt_ranking_num':opt_ranking_num, 'topo_aware':False}) 
#     opt_recommend_df = opt_recommend_df.append(pd.concat([row, new_data],axis=1),ignore_index=True)    

#After Date [JK]
#over_gpu_mem_df = over_gpu_mem_df.tail(3)


#test_raw_df = pd.concat([recommend_df, over_gpu_mem_df])
test_raw_df = recommend_df
test_jobsubmit_df = test_raw_df

#After Date [JK]
#opt_over_gpu_mem_df = pd.DataFrame()
#over_gpu_mem_row_size = over_gpu_mem_df.shape[0]

#for i in range(0,over_gpu_mem_row_size,1):
#    row_data = pd.DataFrame(over_gpu_mem_df.loc[i,:])

#
#After Date [JK]
# for index, row in over_gpu_mem_df.iterrows():
#     row_data = pd.DataFrame(row)
#     row_data = row_data.transpose()
#     row_data['opt_ranking_num'] = [-1]
#     row_data['topo_aware'] = ['False']
#     opt_over_gpu_mem_df = opt_over_gpu_mem_df.append(row_data,ignore_index=True)

# opt_over_gpu_mem_df = pd.DataFrame()
# for row in over_gpu_mem_df.itertuples():
#     new_data = pd.DataFrame({'opt_ranking_num':opt_ranking_num, 'topo_aware':False})
#     opt_over_gpu_mem_df = opt_over_gpu_mem_df.append(pd.concat([row, new_data],axis=1),ignore_index=True)

#After Date [JK]    
#opt_test_jobsubmit_df = pd.concat([opt_recommend_df, opt_over_gpu_mem_df])
opt_test_jobsubmit_df = opt_recommend_df

#test_raw_df = pd.concat([recommend_df, over_gpu_mem_df])
#test_jobsubmit_df = test_raw_df[test_raw_df.columns[:-12]]

pd.options.display.max_columns = None

print(df)
print(sort_df)
print(check_gpu_mem_df)
print(sort_check_gpu_mem_df)
print(recommend_df)
#After Date [JK] 
#print(over_gpu_mem_df)

print(test_jobsubmit_df)

write_file_str = args.write_file_path + '/' + args.write_file

df.to_csv(write_file_str+'.csv', float_format='%.2f', index=False, encoding='utf-8')
sort_df.to_csv(write_file_str+'_sort.csv', float_format='%.2f', index=False, encoding='utf-8')
check_gpu_mem_df.to_csv(write_file_str+'_check_gpu_mem.csv', float_format='%.2f', index=False, encoding='utf-8')
sort_check_gpu_mem_df.to_csv(write_file_str+'_sort_check_gpu_mem.csv', float_format='%.2f', index=False, encoding='utf-8')
recommend_df.to_csv(write_file_str+'_recommend.csv', float_format='%.2f', index=False, encoding='utf-8')

#After Date [JK] 
#over_gpu_mem_df.to_csv(write_file_str+'_over_gpu_mem.csv', float_format='%.2f', index=False, encoding='utf-8')

test_raw_df.to_csv(write_file_str+'_test.csv', float_format='%.2f', index=False, encoding='utf-8')
test_jobsubmit_df.to_csv(write_file_str+'_jobsubmit_test.csv', float_format='%.2f', index=False, encoding='utf-8')
opt_test_jobsubmit_df.to_csv(write_file_str+'_opt_jobsubmit_test.csv', float_format='%.2f', index=False, encoding='utf-8')

opt_head_amp_df.to_csv(write_file_str+'_opt_amp_jobsubmit_test.csv', float_format='%.2f', index=False, encoding='utf-8')

merge_path = "/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/merge4/"
job_script_program = merge_path+ "csv_to_merge_s2_test_total_v2.py"

run_job_file_str = write_file_str + ".sub"
#gen_job_script_cmd = "python" + " " + job_script_program + " " + "--csv_file" + " " + write_file_str + "_jobsubmit_test.csv" + " " + "--run_job_file" + " " + run_job_file_str

gen_job_script_cmd = "python" + " " + job_script_program + " " + "--csv_file" + " " + write_file_str + "_opt_jobsubmit_test.csv" + " " + "--run_job_file" + " " + run_job_file_str

print(gen_job_script_cmd)
os.system(gen_job_script_cmd)



run_amp_job_file_str = write_file_str + "_amp.sub"
#gen_job_script_cmd = "python" + " " + job_script_program + " " + "--csv_file" + " " + write_file_str + "_jobsubmit_test.csv" + " " + "--run_job_file" + " " + run_job_file_str

gen_amp_job_script_cmd = "python" + " " + job_script_program + " " + "--csv_file" + " " + write_file_str + "_opt_amp_jobsubmit_test.csv" + " " + "--run_job_file" + " " + run_amp_job_file_str

print(gen_amp_job_script_cmd)
os.system(gen_amp_job_script_cmd)
#/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/script/final-script-generator 
#gen_job_script
#job_script_program = "csv_to_merge_s2_test.py"

#gen_job_script_cmd = "python" + " " + job_script_program + " " + "--csv_file" + " " + args.write_file + "_jobsubmit_test.csv" + " " + "--run_job_file" + " " + args.run_job_file


#print(gen_job_script_cmd)
#os.system(gen_job_script_cmd)
#python job_script_program --csv_file args.write_file 



