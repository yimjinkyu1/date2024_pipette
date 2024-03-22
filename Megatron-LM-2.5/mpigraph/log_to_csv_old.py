import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd 
import numpy as np 

import datetime 

from tqdm import tqdm 
tqdm.pandas() 

import itertools # permutation 
import pickle
import json
import argparse

import Combination_check_v2 as comb

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--send_file',type=str, default=None)
    parser.add_argument('--recv_file',type=str, default=None)
    parser.add_argument('--write_file',type=str, default=None)
    parser.add_argument('--node_size',type=int, default=None)
    recv_file = parser.parse_args().recv_file
    send_file = parser.parse_args().send_file
    write_file = parser.parse_args().write_file
    node_size = parser.parse_args().node_size

    return recv_file, send_file, write_file, node_size


def main(recv_file, send_file, write_file, node_size):
    tmp = dict()

    tmp_recv = dict()
    data_recv = comb.dataload(recv_file, node_size)
    data_recv, data_recv_col = comb.data_gpu_agg(data_recv, node_size, p2p='Recv')
    tmp_recv["gpus"] = list(data_recv_col) 
    tmp_recv["bandwidth"] = data_recv.tolist()

    # SEND 
    # tmp_send = dict()
    # data_send = comb.dataload(send_file, node_size)
    # data_send, data_send_col = comb.data_gpu_agg(data_send, node_size, p2p='Send')    
    # tmp_send["gpus"] = list(data_send_col) 
    # tmp_send["bandwidth"] = data_send.tolist()        
    tmp[0] = tmp_recv
    #tmp[1] = tmp_send

    with open(write_file, "w") as json_file:
        json.dump(tmp, json_file)    

if __name__ == '__main__':
    recv_file, send_file, write_file, node_size = get_arguments()
    main(recv_file, send_file, write_file, node_size)
    


