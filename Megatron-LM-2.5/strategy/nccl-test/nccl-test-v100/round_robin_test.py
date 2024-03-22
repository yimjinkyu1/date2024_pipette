import os
import json
import time
import argparse
import numpy as np
from subprocess import Popen

def get_arguments():
    parser = argparse.ArgumentParser(description='arguments for nccl test')
    
    parser.add_argument('--run_script', required=True, type=str, help='p2p nccl test run script')
    parser.add_argument('--hostlist', required=True, type=str, help='put your hostlist')
    parser.add_argument('--output_file', required=True, type=str, help='put your output file path')
    parser.add_argument('--data_path', required=True, type=str, help='put your output file path')
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = get_arguments()
    start = time.time()
    data_path = args.data_path

    host_file_name = args.hostlist
    program_nccl_script = args.run_script
    output_file = args.output_file

    with open(host_file_name, 'r') as f: 
        hostfile = f.read().strip('\n').split('\n')
        
    hostindices = np.arange(len(hostfile)).tolist()
    hostdict = {i:hostfile[i] for i in hostindices}
    print(hostdict)

    hostlen = len(hostindices)
    parallel = hostnum = hostlen // 2
    hostlists = [[hostindices]]
    arr = []

    while hostnum > 0:
        hosttmp = []
        for hosts in hostlists:
            hosttmp.extend([[host[:hostnum], host[hostnum:]] for host in hosts])
        hostlists = hosttmp

        hostcomb = []
        for hosts in hostlists:
            tmp = []
            for i in range(hostnum):
                comb = hosts[1][i:] + hosts[1][:i]
                tmp.append([(hosts[0][j], comb[j]) for j in range(hostnum)])
            hostcomb.append(tmp)

        # using half nodes for parallel
        mergenum = parallel // hostnum
        hostsequence = [[] for _ in range(hostnum)]
        for i in range(hostnum):
            hostsequence[i] = [x[i] for x in hostcomb]
        
        hostsequence = [sum(x, []) for x in hostsequence]
        arr.extend(hostsequence)
        hostnum //= 2
        
    print("###################################################################")
    # run nccl test in parallel
    # [yr9.choi@vngpuc8072 nccl-test-rrb]$ sh run_nccl_test.sh "vngpuc8098\nvngpuc8103"

    # Difference node test in parallel
    for nodes in arr:
        if len(set(sum([list(x) for x in nodes], []))) != hostlen: 
            raise ValueError("Inappropriate node combinations")
        
        procs = [Popen(["sh", program_nccl_script, f"{hostdict[i[0]]}\\n{hostdict[i[1]]}", data_path]) for i in nodes]
        print(procs)
        
        for p in procs:
            p.wait()

        time.sleep(3)

    # Same node test in parallel
    procs = [Popen(["sh", program_nccl_script, f"{hostdict[i]}\\n{hostdict[i]}", data_path]) for i in hostindices]
    for p in procs:
        p.wait()
    
    print("###################################################################")

    import itertools
    flatten_arr = list(itertools.chain(*arr)) + [(x, x) for x in hostindices]
    # print(flatten_arr)
    
    nccl_matrix = {}
    for hosts in flatten_arr:
        nccl_log_name = str(data_path) + "/nccl_log_file_" + hostdict[hosts[0]] + "_" + hostdict[hosts[1]]

        with open(nccl_log_name, 'r') as f: 
            nccl_log = f.read().strip('\n').split('\n')

        log_idx = 0
        for idx, log in enumerate(nccl_log):
            if "GB/s" in log:
                log_idx = idx + 1    
                log = nccl_log[log_idx]
                break

        bw = log.strip().split()[7]

        nccl_matrix[hosts[0], hosts[1]] = bw
        nccl_matrix[hosts[1], hosts[0]] = bw
    
    # print(nccl_matrix)     
    
    print("###################################################################")

    # make nccl json file
    json_object = {
        "gpus": list(hostdict.values()),
        "bandwidth": [[ nccl_matrix[i, j] for j in range(hostlen)] for i in range(hostlen)]
    }

    print(json_object)

    with open(output_file, 'w') as f:
        json.dump(json_object, f, indent=2)
        
    print(f"Total Time: {np.round(time.time() - start, 2)}s")
