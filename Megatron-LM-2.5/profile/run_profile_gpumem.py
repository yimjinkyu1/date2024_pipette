import nvidia_smi

import pandas as pd 
import numpy as np 
import torch
import itertools 
import argparse

import time

import json


nvidia_smi.nvmlInit()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file',type=str, default=None)
    parser.add_argument('--time_inter',type=int, default=1)
    output_file = parser.parse_args().output_file
    time_inter = parser.parse_args().time_inter

    return output_file, time_inter

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

def main(output_file, time_inter) : 
    info_dict = {}
    check_proc = 0
    mem_maxUsed = 0 
    while True:
        handle_0 = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        proc_0 = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle_0)
        if proc_0:
            print("ok checked gpu process")
            if check_proc == 0:
                check_proc = 1
            for dev_id in range(nvidia_smi.nvmlDeviceGetCount()):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(dev_id)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

                gpu_info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                gpu_ut = gpu_info.gpu
                #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
                memInfo = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                mem_total = memInfo.total / 1024 / 1024 #MG of memory 
                mem_used = memInfo.used / 1024 /1024 
                mem_free = mem_total - mem_used 
                
                #print(f"Deivce{dev_id}, mem_total : {mem_total},  mem_used : {mem_used}, mem_free : {mem_free}")
                Device_UT = 'Device_' + str(dev_id) +'_UT'
                if Device_UT not in info_dict:
                    info_dict[Device_UT] = [gpu_ut]
                else:
                    info_dict[Device_UT].append(gpu_ut)
                                    
                Device_memUsed = 'Device_'+ str(dev_id) +'_memUsed'
                if Device_memUsed not in info_dict:
                    info_dict[Device_memUsed] = [mem_used]
                else:
                    info_dict[Device_memUsed].append(mem_used)
                
                Device_memFree = 'Device_'+ str(dev_id) +'_memFree'   
                if Device_memFree not in info_dict:
                    info_dict[Device_memFree] = [mem_free]
                else:
                    info_dict[Device_memFree].append(mem_free)   
                     
        else:
            if check_proc == 1:
                save_experiment_information_dict_to_json(info_dict, output_file)  
                exit()
        time.sleep(time_inter)

if __name__ == '__main__': 
    output_file,time_inter = get_arguments()
    main(output_file, time_inter )



    
    
    
