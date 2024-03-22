import pandas as pd
import subprocess

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file',type=str, default=None)
    parser.add_argument('--run_job_file',type=str, default=None)
    
    print(f'csv file name : {parser.parse_args().csv_file}')
    print(f'submit s2 file name : {parser.parse_args().run_job_file}')
    
    
    csv_file_path = "{}".format(parser.parse_args().csv_file)
    run_job_file = "{}".format(parser.parse_args().run_job_file)
    
    #df = pd.read_csv('./parallel-way-64gpu-test10.csv')
    df = pd.read_csv(csv_file_path)
    for i in df.index:
        print(df.loc[i, 'exp_id'], ' is going to be sub file :) I want to be accepted in DATE 2024 lol...\n')
        print(i)
        subprocess.run(' '.join(['python3', '/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/merge4/maker_merge_s2_test_total_v2.py', \
            '--q', df.loc[i, 'q'], \
            '--gpus', str(int(df.loc[i, 'gpus'])), \
            '--fp_type', df.loc[i,'fp_type'], \
            '--exp_id', str(int(df.loc[i, 'exp_id'])), \
            '--sub_test_id', str(int(df.loc[i, 'sub_test_id'])), \
            '--inner_sub_test_id', str(i), \
            '--p2p_dummy_size', str(int(df.loc[i, 'p2p_dummy_size'])), \
            '--model_type', df.loc[i, 'model_type'], \
            '--global_batch_size', str(int(df.loc[i, 'global_batch_size'])), \
            '--micro_batch_size', str(int(df.loc[i, 'micro_batch_size'])), \
            '--parameter_num', df.loc[i, 'parameter_num'], \
            '--tp', str(int(df.loc[i, 'tp'])), \
            '--pp', str(int(df.loc[i, 'pp'])), \
            '--dp', str(int(df.loc[i, 'dp'])), \
            '--num_layers', str(int(df.loc[i, 'num_layers'])), \
            '--hidden_size', str(int(df.loc[i, 'hidden_size'])), \
            '--num_attention_heads', str(int(df.loc[i, 'num_attention_heads'])), \
            '--ddp_impl', df.loc[i,'ddp_impl'], \
            '--train_iters', str(int(df.loc[i, 'train_iters'])), \
            '--network_speed', df.loc[i, 'network_speed'], \
            '--run_job_file', str(run_job_file), \
            '--previous_checkpoint', str(df.loc[i,'previous_checkpoint']), \
            '--topo_aware', str(bool(df.loc[i, 'topo_aware'])), \
            '--nccl_algo', df.loc[i, 'nccl_algo'], \
            '--run_local', df.loc[i, 'run_local'], \
            '--local_server_list', df.loc[i, 'local_server_list'], \
            '--exp_alphabet', df.loc[i, 'exp_alphabet'], \
            '--log_path', df.loc[i, 'log_path'], \
            '--fine_grained_algo', df.loc[i, 'fine_grained_algo'], \
            '--b2_jobid', str(int(df.loc[i, 'b2_jobid'])), \
            '--opt_output_file', df.loc[i, 'opt_output_file'], \
            '--opt_ranking_num', str(int(df.loc[i, 'opt_ranking_num'])), \
            '--write_file_str', df.loc[i, 'write_file_str'], \
            '--gpu_type', df.loc[i, 'gpu_type'], \
            ]), shell=True, check=True)

