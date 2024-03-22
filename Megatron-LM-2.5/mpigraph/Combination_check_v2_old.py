import pandas as pd 
import numpy as np 

import itertools 
import argparse

import time


def __init__() : 
    return 


def pp_node_seq(node_size) : 
    pp_range=list(range(0,node_size))
    pp_node_sequence_list = []
    for sequence_num in itertools.permutations(pp_range,node_size):  # node_size=4
        pp_node_sequence_list.append(list(sequence_num))
    return pp_node_sequence_list


def dataload(filename, node_size) : 
    data = pd.read_csv(filename, sep=',')
    print(f'data.shape: {data.shape}')
    print("0")
    jk_temp = data
    print(f'{jk_temp}')
    target_row_num = node_size * 4
    target_col_num = node_size * 4 + 1
    if data.shape != (target_row_num,target_col_num) : 
        unnamed = data.columns[-1]
        data = data.drop(unnamed, axis=1)
    return data 


def data_gpu_agg(data, node_size, p2p='Recv') : 
    data[f'{p2p}_gpu'] = data[p2p].apply(lambda x : x[:8])
    print("1")
    jk_temp = data
    print(f'{jk_temp}')
    
    data = data.drop(p2p, axis=1)
    jk_temp = data
    print("2")
    print(f'{jk_temp}')
    
    data = data.groupby(f'{p2p}_gpu').sum() 
    
    print("3")
    jk_temp = data
    print(f'{jk_temp}')
    
    
    data = data.rename(columns = lambda x : x[:8]) 
    
    print("4")
    jk_temp = data
    print(f'{jk_temp}')

    data = data.T # Transpose  
    data = data.reset_index()

    data = data.groupby('index').sum() # Calculate Sum  
    
    print("5")
    jk_temp = data
    print(f'{jk_temp}')
    data = data.T # Transpose Again 

    # Set index/Column name 
    data.index.name = "Destination" # From 
    data.columns.name = "Source" # To

    print("6")
    print(f'{len(data)}')
    # diagonal value 
    for i in range(len(data)) : 
        for j in range(len(data)) : 
            m = max(data.iloc[i])
            if i==j : 
                data.iloc[i][j] = m

    print("6")
    jk_temp = data
    print(f'{jk_temp}')
    cols = data.columns
    data = np.array(data)
    data = data / (4*4) # (node_size * node_size)
    return data, cols 

