import os
import pandas as pd 
import numpy as np 

from tqdm import tqdm 
tqdm.pandas() 

import json
import argparse
import torch
import csv

def calc_average(data, element_name):
    
    temp_data = data[element_name]
    min_num = np.min(temp_data)
    max_num = np.max(temp_data)
    sum_num = np.sum(temp_data)

    avg_num = (sum_num - min_num - max_num) / (np.size(temp_data) - 2)

    return avg_num


def save_experiment_information_dict_to_json(info_dict, output_file):
    if output_file:
        with open(output_file, 'w') as f:
                json_string = json.dump(remove_spaces(info_dict), f, indent=4, default=json_default)

def json_default(value):
    if isinstance(value, torch.dtype):
        return str(value)

def remove_spaces(obj):
    wanna_add = []
    wanna_rmv = []
    for key in obj.keys():
        new_key = key.replace(" ", "")
        if new_key != key:
            wanna_add.append(new_key)
            wanna_rmv.append(key)
    for key, new_key in zip(wanna_rmv, wanna_add):
        obj[new_key] = obj[key]
        del obj[key]
    return obj



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--read_gpu_file',type=str, default=None)
    parser.add_argument('--read_train_file',type=str, default=None)
    parser.add_argument('--predict_file',type=str, default=None)
    parser.add_argument('--opt_result_file',type=str, default=None)
    parser.add_argument('--write_file',type=str, default=None)

    
    args = parser.parse_args()

    #info_data = dict()

    #info_data["num_layers"] = args.num_layers
    #info_data["hiddne_size"] = args.hidden_size
    #info_data["num_attention_heads"] = args.num_attention_heads
    #info_data["tp"] = args.tp
    #info_data["mbs"] = args.mbs
    #info_data["gbs"] = args.gbs 


    #gpu_mem_info 
    with open(args.read_gpu_file, "r") as f:
        gpu_data = json.load(f)

    gpus_in_server = 8
    gpu_memUsed_sum = 0
    gpu_memUsed_avg = 0
    gpu_memUsed_max = 0
    
    gpu_ut_sum = 0
    gpu_ut_avg = 0
    gpu_ut_max = 0

    for i in range(0,gpus_in_server,1):
        device_mem_str = "Device_" + str(i) + "_memUsed"
        device_memdata_temp = gpu_data[device_mem_str]
        gpu_memUsed_sum += np.average(device_memdata_temp)
        
        if gpu_memUsed_max < np.max(device_memdata_temp):
            gpu_memUsed_max = np.max(device_memdata_temp)
            
        device_ut_str = "Device_" + str(i) + "_UT"
        device_utdata_temp = gpu_data[device_ut_str]
        gpu_ut_sum += np.average(device_utdata_temp)
        
        if gpu_ut_max < np.max(device_utdata_temp):
            gpu_ut_max = np.max(device_utdata_temp)    


    gpu_memUsed_avg = gpu_memUsed_sum / gpus_in_server
    gpu_ut_avg = gpu_ut_sum / gpus_in_server
    
    print(f"gpu_memUsed_avg : {gpu_memUsed_avg}")
    print(f"gpu_memUsed_max : {gpu_memUsed_max}")
    print(f"gpu_ut_avg : {gpu_ut_avg}")
    print(f"gpu_ut_max : {gpu_ut_max}")

    print(args.opt_result_file)
    with open(args.opt_result_file, "r") as f:
        opt_data = json.load(f)
    opt_pipeline_cost = opt_data[0][1]["pipeline_cost"]
    opt_inter_data_parallel_cost = opt_data[0][1]["inter_data_parallel_cost"]
    opt_inner_data_parallel_cost = opt_data[0][1]["inner_data_parallel_cost"]
    opt_total_parallel_cost = opt_data[0][1]["total_parallel_cost"]
    opt_wall_time = opt_data[0][1]["wall_time"]
    
    print(opt_pipeline_cost)
    print(opt_inter_data_parallel_cost)
    print(opt_inner_data_parallel_cost)
    print(opt_total_parallel_cost)
    print(opt_wall_time)
    

    # with open(args.predict_file, "r") as f:
    #     predict_data = csv.DictReader(f)
        
    # filtered_data = [ item for item in predict_data if item['exp_id'] == 503 and item['sub_test_id']==20 and item['opt_ranking_num']==-1]
    
    # for item in filtered_data:
    #     print(item)

        
    

    #train_info
    with open(args.read_train_file, "r") as f:
        train_data = json.load(f)
    exp_id = train_data["exp_id"]   
    sub_test_id = train_data["sub_test_id"]
    b2_jobid = train_data["b2_jobid"]
    gpu_type = train_data["gpu_type"]
    opt_ranking_num = train_data["opt_ranking_num"]
    elapsed_time_per_iteration = train_data["elapsed-time-per-iteration"]
    world_size = train_data["world_size"]
    fp16 = train_data["fp16"]
    num_layers = train_data["num_layers"]
    hidden_size = train_data["hidden_size"]
    num_attention_heads = train_data["num_attention_heads"]
    tp = train_data["tensor_model_parallel_size"]
    pp = train_data["pipeline_model_parallel_size"]
    dp = train_data["data_parallel_size"]
    server_model_layers = num_layers // pp 
    mbs = train_data["micro_batch_size"]
    gbs = train_data["global_batch_size"]
    avg_fwd_comp_time  = calc_average(train_data, "forward-compute_time")
    avg_bwd_comp_time = calc_average(train_data, "backward-compute_time")
    avg_bwd_allreduce_time = calc_average(train_data, "backward-params-all-reduce_time")
    avg_bwd_emb_time = calc_average(train_data, "backward-embedding-all-reduce_time")
    assignment_json_file = train_data["assignment_json_file"]
    
    print(f"exp_id : {exp_id}")
    print(f"sub_test_id : {sub_test_id}")
    print(f"opt_ranking_num : {opt_ranking_num}")
    predict_data = pd.read_csv(args.predict_file)
    filtered_data = predict_data[(predict_data['exp_id'] == exp_id) & (predict_data['sub_test_id']==sub_test_id) & (predict_data['opt_ranking_num']==opt_ranking_num)]
    
    filtered_data = filtered_data[filtered_data.columns[23:]]
    filtered_data = filtered_data.drop('b2_jobid', axis=1)
    filtered_data = filtered_data.drop('opt_ranking_num', axis=1)
    filtered_data = filtered_data.drop('gpu_type', axis=1)

   
    #write json file 

    #file exist
    if os.path.exists(args.write_file):
        #file read
        with open(args.write_file, "r") as f:
            info_data = json.load(f)
        info_data["exp_id"].append(exp_id)
        info_data["sub_test_id"].append(sub_test_id)
        info_data["b2_jobid"].append(b2_jobid)
        info_data["gpu_type"].append(gpu_type)
        info_data["opt_ranking_num"].append(opt_ranking_num)
        info_data["elapsed_time_per_iteration"].append(elapsed_time_per_iteration)
        info_data["world_size"].append(world_size)
        info_data["fp16"].append(fp16)
        info_data["num_layers"].append(num_layers)
        info_data["hidden_size"].append(hidden_size)
        info_data["num_attention_heads"].append(num_attention_heads)
        info_data["tp"].append(tp)
        info_data["pp"].append(pp)
        info_data["dp"].append(dp)
        info_data["server_model_layers"].append(server_model_layers)
        info_data["mbs"].append(mbs)
        info_data["gbs"].append(gbs)
        info_data["gpu_mem_avg"].append(gpu_memUsed_avg)
        info_data["gpu_mem_max"].append(gpu_memUsed_max)
        info_data["gpu_ut_avg"].append(gpu_ut_avg)
        info_data["gpu_ut_max"].append(gpu_ut_max)
        info_data["avg_fwd_comp_time"].append(avg_fwd_comp_time)
        info_data["avg_bwd_comp_time"].append(avg_bwd_comp_time)
        info_data["avg_bwd_allreduce_time"].append(avg_bwd_allreduce_time)
        info_data["avg_bwd_emb_time"].append(avg_bwd_emb_time)
        info_data["assignment_json_file"].append(assignment_json_file)
        for column_name in filtered_data.columns:
            filtered_data_each_value = filtered_data[column_name].values 
            filtered_data_each_value = filtered_data_each_value[0] 
            if (filtered_data_each_value == 0 or pd.isna(filtered_data_each_value)):
                filtered_data_each_value = 0
            info_data[column_name].append(filtered_data_each_value)
            
    else:
        info_data = {}
        info_data["exp_id"] = [exp_id]
        info_data["sub_test_id"] = [sub_test_id]
        info_data["b2_jobid"] = [b2_jobid]
        info_data["gpu_type"] = [gpu_type]
        info_data["opt_ranking_num"] = [opt_ranking_num]
        info_data["elapsed_time_per_iteration"] = [elapsed_time_per_iteration]
        info_data["world_size"] = [world_size]
        info_data["fp16"] = [fp16]
        info_data["num_layers"] = [num_layers]
        info_data["hidden_size"] = [hidden_size]
        info_data["num_attention_heads"] = [num_attention_heads]
        info_data["tp"] = [tp]
        info_data["pp"] = [pp]
        info_data["dp"] = [dp]
        info_data["server_model_layers"] = [server_model_layers]
        info_data["mbs"] = [mbs]
        info_data["gbs"] = [gbs]
        info_data["gpu_mem_avg"] = [gpu_memUsed_avg]
        info_data["gpu_mem_max"] = [gpu_memUsed_max]
        info_data["gpu_ut_avg"] = [gpu_ut_avg]
        info_data["gpu_ut_max"] = [gpu_ut_max]
        info_data["avg_fwd_comp_time"] = [avg_fwd_comp_time]
        info_data["avg_bwd_comp_time"] = [avg_bwd_comp_time]
        info_data["avg_bwd_allreduce_time"] = [avg_bwd_allreduce_time]
        info_data["avg_bwd_emb_time"] = [avg_bwd_emb_time]
        info_data["assignment_json_file"] = [assignment_json_file]
        for column_name in filtered_data.columns:
            filtered_data_each_value = filtered_data[column_name].values 
            filtered_data_each_value = filtered_data_each_value[0]
            if (filtered_data_each_value == 0 or pd.isna(filtered_data_each_value)):
                filtered_data_each_value = 0
            info_data[column_name] = [filtered_data_each_value]
    save_experiment_information_dict_to_json(info_data, args.write_file)
    #with open(args.write_file, "w") as json_file:
    #        json.dump(info_data, json_file)    

    #try:
    #    with open(args.write_file, "r") as f:
    #        info_data = json.load(f)
    #except FileNotFoundError:
    #    info_data = {}


if __name__ == '__main__':
    main()
    


