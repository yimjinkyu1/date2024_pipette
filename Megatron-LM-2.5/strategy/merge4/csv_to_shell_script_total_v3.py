import pandas as pd
import subprocess

import argparse
import os.path
from os import path 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file',type=str, default=None)
    parser.add_argument('--shell_file',type=str, default=None)
    args = parser.parse_args()
    
    print(f'csv file name : {args.csv_file}')
    print(f'shell file name : {args.csv_file}')
    csv_file_path = "./total_submit_file/{}.csv".format(args.csv_file)
    
    #df = pd.read_csv('./parallel-way-64gpu-test10.csv')
    df = pd.read_csv(csv_file_path)

    

    test_id = 0
    for i in df.index:

        exp_id=df.loc[i, 'exp_id']
        gpus = df.loc[i, 'gpus']
        server= df.loc[i, 'server']
        model_name=df.loc[i, 'model_name']
        model_type=df.loc[i, 'model_type']

        gbs=df.loc[i, 'gbs']
        layers=df.loc[i, 'layers']
        hiddens=df.loc[i, 'hiddens']
        heads=df.loc[i, 'heads']
        network_speed=df.loc[i, 'network_speed']
        fp_type=df.loc[i, 'fp_type']
        nccl_algo=df.loc[i, 'nccl_algo']
        
        #0820
        gpu_type=df.loc[i, 'gpu_type']
        pp_inter_bw=df.loc[i,'pp_inter_bw']
        dp_inter_bw=df.loc[i,'dp_inter_bw']
        dp_inner_bw=df.loc[i,'dp_inner_bw']
        
        #0822
        run_local=df.loc[i,'run_local']
        local_server_list=df.loc[i,'local_server_list']
        
        #0823
        nccl_test_file=df.loc[i,'nccl_test_file']

        #828
        exp_alphabet=df.loc[i,'exp_alphabet']
        
        
        #0907 
        pipette = df.loc[i,'pipette']
        tp = df.loc[i,'tp']
        pp = df.loc[i,'pp']
        dp = df.loc[i,'dp']
        mbs = df.loc[i,'mbs']
        
        #240122
        fine_grained_algo = df.loc[i,'fine_grained_algo']
        
        write_file_str = "{}-{}-{}-{}gpu-{}gbs-{}layers-{}hidden-{}head-{}-{}-pipette-{}".format(exp_id, gpu_type, model_name, gpus, gbs, layers, hiddens, heads, nccl_algo, exp_alphabet,pipette)

        
        if i == 0:
            lsf_string = '#! /bin/bash\n'
            lsf_string += \
"""

sh /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/clear.sh

export CC=/apps/gcc/gcc-8.3.0/bin/gcc
export CXX=/apps/gcc/gcc-8.3.0/bin/g++

export OMP_NUM_THREADS=128
export NCCL_IB_TIMEOUT=220
export NCCL_ASYNC_ERROR_HANDLING=0

DATE=$(date +%Y%m%d_%H-%M-%S)
""".format(gpu_type, exp_alphabet)

            data_path = ''
            if model_type == 'GPT':
                data_path = '/scratch/jinkyu.yim/NLP/my-gpt2_text_document'
            else:
                data_path = '/scratch/jinkyu.yim/bert_data/my-bert_text_sentence'
            
            lsf_string += '\n'
            if run_local == 'local':
                lsf_string += \
"""
export CC=/apps/gcc/gcc-8.3.0/bin/gcc
export CXX=/apps/gcc/gcc-8.3.0/bin/g++

export NCCL_IB_TIMEOUT=220
export NCCL_ASYNC_ERROR_HANDLING=0

DATE=$(date +%Y%m%d_%H-%M-%S)
DATA_PATH={}

LSB_JOBID=3333
B2_JOBID=3333

export HOSTLIST=`cat /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/script/{} | sort`

GPUS_PER_NODE=8
MASTER_PORT="12223"
NNODES=`echo $HOSTLIST | wc -w`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
KILL_PROGRAM="/home/sr5/jinkyu.yim/Megatron-CC-main/scripts/kill_garbage_process.sh"


#profile program
PROFILE_GPU_MEM_PROGRAM="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/profile/run_profile_gpumem.py"
MERGE_PROFILE_PROGRAM="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/profile/log_to_json_v2.py"
MERGE_PROFILE_FILE="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/script/total_profile/merge_profile_real_result_{}_v2.json"

""".format(data_path, gpu_type, exp_alphabet, local_server_list, gpu_type)
            else:
                lsf_string += \
"""
export CC=/apps/gcc/gcc-8.3.0/bin/gcc
export CXX=/apps/gcc/gcc-8.3.0/bin/g++

export NCCL_IB_TIMEOUT=220
export NCCL_ASYNC_ERROR_HANDLING=0

DATE=$(date +%Y%m%d_%H-%M-%S)
DATA_PATH={}

B2_JOBID=$B2_JOBID
LSB_JOBID=$B2_JOBID

export HOSTLIST=`echo $B2_DP_HOSTS | sed s/,/' '/g`

GPUS_PER_NODE=8
MASTER_PORT="12223"
NNODES=`echo $HOSTLIST | wc -w`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
KILL_PROGRAM="/home/sr5/jinkyu.yim/Megatron-CC-main/scripts/kill_garbage_process.sh"

#profile program
PROFILE_GPU_MEM_PROGRAM="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/profile/run_profile_gpumem.py"
MERGE_PROFILE_PROGRAM="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/profile/log_to_json_v2.py"
MERGE_PROFILE_FILE="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/script/total_profile/merge_profile_real_result_{}_v2.json"

""".format(data_path, gpu_type)
    
            lsf_string += \
"""
#nccl program
#PROGRAM_NCCL_TEST="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/nccl-test/multi_nccl_test.sh"
PROGRAM_NCCL_TEST="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/nccl-test/nccl-test-{}/run_rrb_nccl_test.sh"

#do-mps
PROGRAM_DO_MPS="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/merge4/llm_train_strategy_assignment_total_v3.py"

PROGRAM_MAKER_MERGE="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/merge4/maker_merge_s2_test_total_other.py"

#log path
LOG_PATH=/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/script/{}-test/{}/{}_expid_{}_${{DATE}}_${{LSB_JOBID}}
""".format(gpu_type, model_type,fp_type, fp_type, exp_id)

            lsf_string += \
"""
CLEAN_GARBAGE_RANK_FILE=${LOG_PATH}/mpirank/garbage_rank_file

#opt_path
OPT_FILE_PATH="${LOG_PATH}/opt/opt_result_file"

mkdir -p ${LOG_PATH}
mkdir -p ${LOG_PATH}/mpirank
mkdir -p ${LOG_PATH}/mpigraph
mkdir -p ${LOG_PATH}/assignment
mkdir -p ${LOG_PATH}/jsonlog/
mkdir -p ${LOG_PATH}/profile/
mkdir -p ${LOG_PATH}/nccl/
mkdir -p ${LOG_PATH}/job/
mkdir -p ${LOG_PATH}/write_file/
mkdir -p ${LOG_PATH}/opt/
"""

            lsf_string += \
"""
NCCL_OUTPUT_FILE="${LOG_PATH}/nccl/nccl_output.json"

#NCCL_OUTPUT_FILE="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/script/GPT-test/fp32/fp32_expid_17_20230908_17-10-48_3333/nccl/nccl_output.json"

RUN_JOB_FILE="${LOG_PATH}/job/run_job_file"

clean_rank_options(){
    for HOST in $HOSTLIST
        do
                echo "${HOST} slots=1" >> ${LOG_PATH}/mpirank/garbage_rank_file
        done
}
#celan garbage gpu process
clean_rank_options
mpirun -n $NNODES -x LD_LIBRARY_PATH -x PATH -hostfile ${LOG_PATH}/mpirank/garbage_rank_file sh ${KILL_PROGRAM}
sleep 1

#sort host file 
for HOST in $HOSTLIST
do
    echo "${HOST}" >> ${LOG_PATH}/mpirank/unsorted_rank_file
done
HOSTLIST=`cat ${LOG_PATH}/mpirank/unsorted_rank_file | sort`
for HOST in $HOSTLIST
do
    echo "${HOST}" >> ${LOG_PATH}/mpirank/sorted_rank_file
done

#nccl run (multi_nccl_test.sh HOST_FILE DATA_PATH OUTPUT_FILE)
start_nccl_time=$(date +%s)
$PROGRAM_NCCL_TEST ${LOG_PATH}/mpirank/sorted_rank_file ${LOG_PATH}/nccl/ $NCCL_OUTPUT_FILE
ps -ef | grep -v grep | grep run_rrb_nccl_test
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep run_rrb_nccl_test
done
end_nccl_time=$(date +%s)
duration_nccl_time=$((end_nccl_time - start_nccl_time))
echo "Execution time nccl : $duration_nccl_time seconds"
"""
        if pipette == "all":
            lsf_string += \
"""
start_st_time=$(date +%s)
RMD=`python ${{PROGRAM_DO_MPS}} --log_path ${{LOG_PATH}} --write_file_path ${{LOG_PATH}}/write_file --write_file {} --exp_id {} --server {} --model_type {} --parameter_num "test" --global_batch_size {} --ddp_impl "local" --fp_type {} --train_iters 10 --network_speed {} --q "admin_gpu" --previous_checkpoint -1 --p2p_dummy_size 1 --num_layers {} --hidden_size {} --num_attention_heads {} --seq_length 1024 --vocab_size 50256 --pp_inter_bw {} --dp_inter_bw {} --dp_inner_bw {} --nccl_test_file ${{NCCL_OUTPUT_FILE}} --sa_initial_temp 1 --sa_alpha 0.999 --sa_time_limit 10 --gpu_type {} --b2_jobid ${{B2_JOBID}}  --exp_alphabet {} --opt_output_file ${{OPT_FILE_PATH}} --nccl_algo RingTree --micro_batch_size {} --pp {} --tp {} --fine_grained_algo {}`
""".format(write_file_str, exp_id, server, model_type, gbs, fp_type, network_speed, layers, hiddens, heads, pp_inter_bw, dp_inter_bw, dp_inner_bw,  gpu_type,  exp_alphabet, mbs, pp, tp, fine_grained_algo)

            lsf_string += \
"""
echo ""
echo ${{RMD}}
ps -ef | grep -v grep | grep llm_train_strategy_assignment_total
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep llm_train_strategy_assignment_total
done

end_st_time=$(date +%s)
duration_st_time=$((end_st_time - start_st_time))
echo "Execution time strategy : $duration_st_time seconds"



RUN_AMP_JOB_FILE_STR="${{LOG_PATH}}/write_file/{}_amp.sub"

sh $RUN_AMP_JOB_FILE_STR

ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
done

mpirun -n $NNODES -x LD_LIBRARY_PATH -x PATH -hostfile ${{LOG_PATH}}/mpirank/garbage_rank_file sh ${{KILL_PROGRAM}}
sleep 1
ps -ef | grep -v grep | grep kill_garbage_process
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep kill_garbage_process
done

RUN_JOB_FILE_STR="${{LOG_PATH}}/write_file/{}.sub"

sh $RUN_JOB_FILE_STR

ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
done

""".format(write_file_str, write_file_str)
        
        if pipette == "amp":
            lsf_string += \
"""
start_st_time=$(date +%s)
RMD=`python ${{PROGRAM_DO_MPS}} --log_path ${{LOG_PATH}} --write_file_path ${{LOG_PATH}}/write_file --write_file {} --exp_id {} --server {} --model_type {} --parameter_num "test" --global_batch_size {} --ddp_impl "local" --fp_type {} --train_iters 10 --network_speed {} --q "admin_gpu" --previous_checkpoint -1 --p2p_dummy_size 1 --num_layers {} --hidden_size {} --num_attention_heads {} --seq_length 1024 --vocab_size 50256 --pp_inter_bw {} --dp_inter_bw {} --dp_inner_bw {} --nccl_test_file ${{NCCL_OUTPUT_FILE}} --sa_initial_temp 1 --sa_alpha 0.999 --sa_time_limit 10 --gpu_type {} --b2_jobid ${{B2_JOBID}}  --exp_alphabet {} --opt_output_file ${{OPT_FILE_PATH}} --nccl_algo RingTree --micro_batch_size {} --pp {} --tp {} --fine_grained_algo {}`
""".format(write_file_str, exp_id, server, model_type, gbs, fp_type, network_speed, layers, hiddens, heads, pp_inter_bw, dp_inter_bw, dp_inner_bw,  gpu_type,  exp_alphabet, mbs, pp, tp, fine_grained_algo)

            lsf_string += \
"""
echo ""
echo ${{RMD}}
ps -ef | grep -v grep | grep llm_train_strategy_assignment_total
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep llm_train_strategy_assignment_total
done
end_st_time=$(date +%s)
duration_st_time=$((end_st_time - start_st_time))
echo "Execution time strategy : $duration_st_time seconds"


RUN_AMP_JOB_FILE_STR="${{LOG_PATH}}/write_file/{}_amp.sub"

sh $RUN_AMP_JOB_FILE_STR

ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
done

""".format(write_file_str, write_file_str)

        if pipette == "our":
            lsf_string += \
"""
start_st_time=$(date +%s)
RMD=`python ${{PROGRAM_DO_MPS}} --log_path ${{LOG_PATH}} --write_file_path ${{LOG_PATH}}/write_file --write_file {} --exp_id {} --server {} --model_type {} --parameter_num "test" --global_batch_size {} --ddp_impl "local" --fp_type {} --train_iters 10 --network_speed {} --q "admin_gpu" --previous_checkpoint -1 --p2p_dummy_size 1 --num_layers {} --hidden_size {} --num_attention_heads {} --seq_length 1024 --vocab_size 50256 --pp_inter_bw {} --dp_inter_bw {} --dp_inner_bw {} --nccl_test_file ${{NCCL_OUTPUT_FILE}} --sa_initial_temp 1 --sa_alpha 0.999 --sa_time_limit 10 --gpu_type {} --b2_jobid ${{B2_JOBID}}  --exp_alphabet {} --opt_output_file ${{OPT_FILE_PATH}} --nccl_algo RingTree --micro_batch_size {} --pp {} --tp {} --fine_grained_algo {}`
""".format(write_file_str, exp_id, server, model_type, gbs, fp_type, network_speed, layers, hiddens, heads, pp_inter_bw, dp_inter_bw, dp_inner_bw,  gpu_type,  exp_alphabet, mbs, pp, tp, fine_grained_algo)

            lsf_string += \
"""
echo ""
echo ${{RMD}}
ps -ef | grep -v grep | grep llm_train_strategy_assignment_total
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep llm_train_strategy_assignment_total
done
end_st_time=$(date +%s)
duration_st_time=$((end_st_time - start_st_time))
echo "Execution time strategy : $duration_st_time seconds"


RUN_JOB_FILE_STR="${{LOG_PATH}}/write_file/{}.sub"

sh $RUN_JOB_FILE_STR

ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
done

""".format(write_file_str, write_file_str)


        if pipette == "no":
            pass
            RUN_JOB_FILE="${{LOG_PATH}}/write_file/{}.sub".format(write_file_str)
            lsf_string += \
"""
RMD=`python ${{PROGRAM_MAKER_MERGE}} --q "admin_gpu" --gpus {} --fp_type {} --exp_id {} --sub_test_id 0 --inner_sub_test_id 0 --p2p_dummy_size 1 --model_type {} --global_batch_size {} --micro_batch_size {} --parameter_num "test" --tp {} --pp {} --dp {} --num_layers {} --hidden_size {} --num_attention_heads {} --ddp_impl "local" --train_iters 10 --network_speed {} --run_job_file {} --previous_checkpoint -1 --topo_aware "False" --nccl_algo RingTree --run_local {} --local_server_list {} --exp_alphabet {} --log_path ${{LOG_PATH}} --b2_jobid ${{B2_JOBID}} --gpu_type {}`
""".format(gpus, fp_type, exp_id,model_type, gbs, mbs, tp, pp, dp, layers, hiddens, heads, network_speed, RUN_JOB_FILE, run_local, local_server_list, exp_alphabet, gpu_type )

            lsf_string += \
"""
RUN_JOB_FILE_STR="${{LOG_PATH}}/write_file/{}.sub"

sh $RUN_JOB_FILE_STR
""".format(write_file_str)
            lsf_string += \
"""
ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep sub | grep gpu | grep -v merge
done
"""
        lsf_string += \
"""
# {}) {}-{}-{}-{}gpu-{}gbs-{}layers-{}hidden-{}head-{}-{}\n

mpirun -n $NNODES -x LD_LIBRARY_PATH -x PATH -hostfile ${{LOG_PATH}}/mpirank/garbage_rank_file sh ${{KILL_PROGRAM}}
sleep 1
ps -ef | grep -v grep | grep kill_garbage_process
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep kill_garbage_process
done
""".format(i, exp_id, gpu_type, model_name, gpus, gbs, layers, hiddens, heads, nccl_algo, exp_alphabet)

        print(i)

    shell_file_path = "./total_submit_file/{}_{}_submit.sub".format(args.csv_file,exp_alphabet)
    print(f'shell file name : {shell_file_path}')
    if path.isfile(shell_file_path):
        os.remove(shell_file_path)

    f = open(shell_file_path , 'a')
    f.write(lsf_string)
    f.close()    
