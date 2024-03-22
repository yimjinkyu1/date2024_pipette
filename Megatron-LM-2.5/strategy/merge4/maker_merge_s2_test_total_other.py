
#!/usr/bin/python3

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lsf script maker')

    # add arguments
    parser.add_argument('--q', default='admin_dev_gpu_all', type=str, required=False, help='lsf gpu queue.')
    parser.add_argument('--gpus', default=128, type=int, required=True, help='num of gpus to use.')
    parser.add_argument('--exp_id', required=True, type=int, help='put your experiment id.')
    parser.add_argument('--model_type', required=True, type=str, help='put your model type.')
    parser.add_argument('--parameter_num', required=True, type=str, help='put your parameter num')
    parser.add_argument('--tp', required=True, type=int, help='tensor parallel size')
    parser.add_argument('--pp', required=True, type=int, help='pipeline parallel size')
    parser.add_argument('--dp', required=True, type=int, help='data parallel size')
    parser.add_argument('--num_layers', required=False, type=int, help='number of train model layer')
    parser.add_argument('--hidden_size', required=False, type=int, help='hidden size of transformer model')
    parser.add_argument('--num_attention_heads', required=False, type=int, help='attention node size of transformer model')
    parser.add_argument('--train_iters', required=True, type=int, help="the numbere of training iters")
    parser.add_argument('--network_speed', required=True, type=str, help="200G or 800G")
    parser.add_argument('--sub_test_id', required=True, type=int, help='put your test id.')
    parser.add_argument('--inner_sub_test_id', required=True, type=int, help='put your inner test id.')
    parser.add_argument('--global_batch_size', required=False, type=int, default=512, help='global batch size')
    parser.add_argument('--micro_batch_size', required=False, type=int, default=8, help='micro batch size')
    parser.add_argument('--ddp_impl', required=True, type=str, help='put your ddp-impl option.')
    parser.add_argument('--fp_type', required=True, type=str, help='put your floating-point type.(fp32 or fp16 or df16)')
    parser.add_argument('--previous_checkpoint', required=True, type=str, help='Is there a previous checkpoint ?')
    parser.add_argument('--topo_aware', default=True, type=str, help='use network topo aware & latency information for layer stage assignment')
    parser.add_argument('--nccl_algo', default="RingTree", type=str, help='use nccl algorithm')
    parser.add_argument('--p2p_dummy_size', default=1, type=int, required=True, help='p2p_dummy_size.')
    parser.add_argument('--run_job_file', type=str, required=True, help='submit scheducler job file')
    parser.add_argument('--run_local', type=str, required=False, default="s2", help='submit scheducler job file(s2) or local server(local)')
    parser.add_argument('--local_server_list', type=str, required=False, default="1102_32gpu", help='local server list')
    parser.add_argument('--exp_alphabet', type=str, required=False, default="a", help='experiment alphabet(a,b,c,...')
    parser.add_argument('--opt_output_file', type=str, required=False, help='simulated annealing partition')
    parser.add_argument('--gpu_type', type=str, required=False, default="a100", help='V100 or A100')
    parser.add_argument('--log_path', type=str, required=True, help='log path')
    parser.add_argument('--b2_jobid', default=5555, type=int, help='scheduler job id (s2)')
    parser.add_argument('--opt_ranking_num', default=1, type=int, help='optimizer ranking num')
    
    args = parser.parse_args()
    
    if args.nccl_algo == "RingTree":
        args.nccl_algo = "Ring,Tree"
    elif args.nccl_algo == "RingCollNet":
        args.nccl_algo = "Ring,CollNet"
    elif args.nccl_algo == "TreeCollNet":
        args.nccl_algo = "Tree,CollNet"
    #
    if args.model_type == "GPT":
        pre_run_script = "pretrain_gpt.py"
    elif args.model_type == "BERT":
        pre_run_script = "pretrain_bert.py"
    else:
        print("Unkown model type (GPT,BERT)")
        
    lsf_string = ''
    output_base_path = '/scratch/jinkyu.yim/checkpoint/vldb'     
    

    
    RANKFILE_PATH= '${LOG_PATH}/mpirank'
    UNSORTED_RANK_FILE='{}/unsorted_rank_file'.format(RANKFILE_PATH)
    SORTED_RANK_FILE='{}/sorted_rank_file'.format(RANKFILE_PATH)
    CLEAN_GARBAGE_RANK_FILE='{}/garbage_rank_file'.format(RANKFILE_PATH)
    
    MPIGRAPH_LOG_PATH='${LOG_PATH}/mpigraph'
    MPIGRAPH_RANK_FILE='{}/mpigraph_rank_file'.format(MPIGRAPH_LOG_PATH)
    MPIGRAPH_LOG_FILE='{}/mpigraph_log_file'.format(MPIGRAPH_LOG_PATH)
    MPIGRAPH_JSON_FILE='{}/mpigraph_file.json'.format(MPIGRAPH_LOG_PATH)

    JSON_LOG_PATH='${LOG_PATH}/jsonlog/'
    PROFILE_LOG_PATH='${LOG_PATH}/profile/'
    MERGE_LOG_PATH='${LOG_PATH}/merge'
    ASSIGNMENT_PATH='${LOG_PATH}/assignment'

    PP_FACTORIAL_FILE='{}/pp_factorial.json'.format(ASSIGNMENT_PATH)
    PP_GLS_FILE='{}/pp_gls.json'.format(ASSIGNMENT_PATH)
    PP_SIMUANN_FILE='{}/pp_simuann.json'.format(ASSIGNMENT_PATH)
    PP_MHCP_FILE='{}/pp_mhcp.json'.format(ASSIGNMENT_PATH)
    PARALLEL_GROUP_FILE='{}/parallel_group.json'.format(ASSIGNMENT_PATH)
    
    if args.inner_sub_test_id == 0:
        lsf_string += '#! /bin/bash\n'
        lsf_string += '\n'  
        lsf_string += \
"""
LOG_PATH={}
""".format(args.log_path)

        lsf_string += \
"""
sh /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/clear.sh\n
""".format(args.gpu_type, args.exp_alphabet)
        
        data_path = ''
        if args.model_type == 'GPT':
            data_path = '/scratch/jinkyu.yim/NLP/my-gpt2_text_document'
        else:
            data_path = '/scratch/jinkyu.yim/bert_data/my-bert_text_sentence'
        
        if args.run_local == 'local':
            lsf_string += \
"""
export CC=/apps/gcc/gcc-8.3.0/bin/gcc
export CXX=/apps/gcc/gcc-8.3.0/bin/g++

export NCCL_IB_TIMEOUT=220
export NCCL_ASYNC_ERROR_HANDLING=0

DATE=$(date +%Y%m%d_%H-%M-%S)
DATA_PATH={}

LSB_JOBID=3333

export HOSTLIST=`cat /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/script/{} | sort`

GPUS_PER_NODE=8
MASTER_PORT="12223"
NNODES=`echo $HOSTLIST | wc -w`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
KILL_PROGRAM="/home/sr5/jinkyu.yim/Megatron-CC-main/scripts/kill_garbage_process.sh"


#profile program
PROFILE_GPU_MEM_PROGRAM="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/profile/run_profile_gpumem.py"
MERGE_PROFILE_PROGRAM="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/profile/log_to_json.py"
MERGE_PROFILE_FILE="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/script/total_profile/merge_profile_real_result_{}.json"

""".format(data_path,args.gpu_type, args.exp_alphabet, args.local_server_list, args.gpu_type)
        else:
            lsf_string += \
"""
export CC=/apps/gcc/gcc-8.3.0/bin/gcc
export CXX=/apps/gcc/gcc-8.3.0/bin/g++

export OMP_NUM_THREADS=64
export NCCL_IB_TIMEOUT=220
export NCCL_ASYNC_ERROR_HANDLING=0

DATE=$(date +%Y%m%d_%H-%M-%S)
DATA_PATH={}

B2_JOBID={}
LSB_JOBID={}

export HOSTLIST=`cat {}/mpirank/sorted_rank_file | sort`

GPUS_PER_NODE=8
MASTER_PORT="12223"
NNODES=`echo $HOSTLIST | wc -w`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
KILL_PROGRAM="/home/sr5/jinkyu.yim/Megatron-CC-main/scripts/kill_garbage_process.sh"


#profile program
PROFILE_GPU_MEM_PROGRAM="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/profile/run_profile_gpumem.py"
MERGE_PROFILE_PROGRAM="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/profile/log_to_json.py"
MERGE_PROFILE_FILE="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/script/total_profile/merge_profile_real_result_{}.json"

""".format(data_path, args.b2_jobid, args.b2_jobid, args.log_path, args.gpu_type )
            

        lsf_string += \
"""
CLEAN_GARBAGE_RANK_FILE={}
""".format(CLEAN_GARBAGE_RANK_FILE)

        lsf_string += \
"""
mpirun -n $NNODES -x LD_LIBRARY_PATH -x PATH -hostfile {} sh ${{KILL_PROGRAM}}
sleep 2
""".format(CLEAN_GARBAGE_RANK_FILE)

        lsf_string += '\n'

    if (args.model_type == "GPT") and (args.parameter_num == "2.5B"):
        num_layers = 52
        hidden_size = 1920
        num_attention_heads = 24
        seq_length = 1024
        max_position_embeddings = 1024
        
    elif (args.model_type == "GPT") and (args.parameter_num == "2.7B"):
        num_layers = 56
        hidden_size = 1920
        num_attention_heads = 24
        seq_length = 1024
        max_position_embeddings = 1024
        
    elif (args.model_type == "GPT") and (args.parameter_num == "3.2B"):
        num_layers = 64
        hidden_size = 1920
        num_attention_heads = 24
        seq_length = 1024
        max_position_embeddings = 1024


    elif (args.model_type == "GPT") and (args.parameter_num == "335M"):
        num_layers = 24
        hidden_size = 1024
        num_attention_heads = 16
        seq_length = 1024
        max_position_embeddings = 1024
        
    elif (args.model_type == "GPT") and (args.parameter_num == "5.9B"):
        num_layers = 32
        hidden_size = 3840
        num_attention_heads = 32
        seq_length = 1024
        max_position_embeddings = 1024
      
    elif (args.model_type == "GPT") and (args.parameter_num == "8.1B"):
        num_layers = 64
        hidden_size = 3072
        num_attention_heads = 24
        seq_length = 1024
        max_position_embeddings = 1024
        
    elif (args.model_type == "GPT") and (args.parameter_num == "8.3B"):
        num_layers = 72
        hidden_size = 3072
        num_attention_heads = 24
        seq_length = 1024
        max_position_embeddings = 1024

    elif (args.model_type == "GPT") and (args.parameter_num == "18B"):
        num_layers = 40
        hidden_size = 6144
        num_attention_heads = 48
        seq_length = 1024
        max_position_embeddings = 1024
        
    elif (args.model_type == "GPT") and (args.parameter_num == "21B"):
        num_layers = 48
        hidden_size = 6144
        num_attention_heads = 48
        seq_length = 1024
        max_position_embeddings = 1024


    elif (args.model_type == "GPT") and (args.parameter_num == "39B"):
        num_layers = 48
        hidden_size = 8192
        num_attention_heads = 64
        seq_length = 1024
        max_position_embeddings = 1024
        
    elif (args.model_type == "GPT") and (args.parameter_num == "175B"):
        num_layers = 96
        hidden_size = 12288
        num_attention_heads = 96
        seq_length = 1024
        max_position_embeddings = 1024

    elif (args.model_type == "BERT") and (args.parameter_num == "1.3B"):
        num_layers = 24
        hidden_size = 2048
        num_attention_heads = 32
        seq_length = 512
        max_position_embeddings = 512
       
    elif (args.model_type == "BERT") and (args.parameter_num == "3.9B"):
        num_layers = 48
        hidden_size = 2560
        num_attention_heads = 40
        seq_length = 512
        max_position_embeddings = 512 
    elif (args.model_type == "GPT") and ((args.parameter_num == "profile") or (args.parameter_num == "test")):
        num_layers = args.num_layers
        hidden_size = args.hidden_size
        num_attention_heads = args.num_attention_heads
        seq_length = 1024
        max_position_embeddings = 1024
    else :
        print('Unknown model or Unknown parameter number')
        exit()
        

    lsf_string += '\n'
    exp_name = 'SC-{}-{}-{}-${{NNODES}}server_{}bps_{}tp_{}pp_${{DATA_PARALLEL_SIZE}}dp_{}mibatch_{}globatch_${{B2_JOBID}}_expid_{}_subtestID_{}_optrankingNum_{}'.format(args.model_type, args.parameter_num, args.fp_type, args.network_speed, args.tp, args.pp, args.micro_batch_size, args.global_batch_size, args.exp_id, args.sub_test_id,args.opt_ranking_num)
    lsf_string += \
"""
#exp_id : {}, subtetsID: {}
""".format(args.exp_id, args.sub_test_id)

    
    lsf_string += '\n'

    if args.previous_checkpoint == '-1':
        lsf_string += '#chekpoint path'
        lsf_string += '\n'
        lsf_string += 'mkdir -p {}/expid_{}_jobid_${{B2_JOBID}}_${{DATE}}\n'.format(output_base_path, args.exp_id)
        lsf_string += 'CHECKPOINT_PATH={}/{}_${{DATE}}\n'.format(output_base_path, exp_name)

    elif args.previous_checkpoint == '-1.0':
        lsf_string += '#chekpoint path'
        lsf_string += '\n'
        lsf_string += 'mkdir -p {}/expid_{}_jobid_${{B2_JOBID}}_${{DATE}}\n'.format(output_base_path, args.exp_id)
        lsf_string += 'CHECKPOINT_PATH={}/{}_${{DATE}}\n'.format(output_base_path, exp_name)

    else :
        lsf_string+='CHECKPOINT_PATH={}\n'.format(args.previous_checkpoint)
        
    lsf_string += 'echo \"checkpoint path : ${CHECKPOINT_PATH}\"\n'
    lsf_string += '\n\n'
    lsf_string += \
"""
TENSOR_MP_SIZE={}
PIPELINE_MP_SIZE={}
DATA_PARALLEL_SIZE=`expr $WORLD_SIZE / $PIPELINE_MP_SIZE / $TENSOR_MP_SIZE`
MICRO_BATCH_SIZE={}
GLOBAL_BATCH_SIZE={}
FP_TYPE={}
""".format(args.tp, args.pp, args.micro_batch_size, args.global_batch_size, args.fp_type)

    lsf_string += 'echo \"${TENSOR_MP_SIZE}tp_${PIPELINE_MP_SIZE}pp_${DATA_PARALLEL_SIZE}dp\"\n\n\n'

    lsf_string += '\n\n'
    
    if args.model_type == "GPT":
        lsf_string += \
"""
TOTAL_ARGS="--num-layers {} \\
            --hidden-size {} \\
            --num-attention-heads {} \\
            --micro-batch-size {} \\
            --global-batch-size {} \\
            --seq-length {} \\
            --max-position-embeddings {} \\
            --train-iters {} \\
            --lr-decay-iters 320000 \\
            --save $CHECKPOINT_PATH \\
            --load $CHECKPOINT_PATH \\
            --data-path $DATA_PATH \\
            --vocab-file gpt2-vocab.json \\
            --merge-file gpt2-merges.txt \\
            --data-impl mmap \\
            --split 949,50,1 \\
            --distributed-backend nccl \\
            --lr 0.00015 \\
            --lr-decay-style cosine \\
            --min-lr 1.0e-5 \\
            --weight-decay 1e-2 \\
            --clip-grad 1.0 \\
            --lr-warmup-fraction .01 \\
            --log-interval 1 \\
            --save-interval 50 \\
            --eval-interval 50 \\
            --eval-iters 10 \\
            --tensor-model-parallel-size $TENSOR_MP_SIZE \\
            --pipeline-model-parallel-size $PIPELINE_MP_SIZE \\
            --DDP-impl {} \\
            --mpigraph_log_file {} \\
            --experiment_name {} \\
            --save_json_log \\
            --save_json_path {} \\
            --exp_id {} \\
            --sub_test_id {} \\
            --b2_jobid {} \\
            --gpu_type {} \\
            --p2p_dummy_size {} \\
            --assignment_json_file {} \\
            --opt_ranking_num {} \\
            --no-masked-softmax-fusion \\
            --no-bias-dropout-fusion \\
            --no-bias-gelu-fusion \\
""".format(num_layers, hidden_size, num_attention_heads, args.micro_batch_size, args.global_batch_size, seq_length, max_position_embeddings, args.train_iters,args.ddp_impl, MPIGRAPH_JSON_FILE, exp_name, JSON_LOG_PATH, args.exp_id, args.sub_test_id, args.b2_jobid, args.gpu_type,args.p2p_dummy_size, args.opt_output_file, args.opt_ranking_num)
    elif args.model_type == "BERT":
        lsf_string += \
"""
TOTAL_ARGS="--num-layers {} \\
            --hidden-size {} \\
            --num-attention-heads {} \\
            --micro-batch-size {} \\
            --global-batch-size {} \\
            --seq-length {} \\
            --max-position-embeddings {} \\
            --train-iters {} \\
            --lr-decay-iters 990000 \\
            --save $CHECKPOINT_PATH \\
            --load $CHECKPOINT_PATH \\
            --data-path $DATA_PATH \\
            --vocab-file bert-large-uncased-vocab.txt \\
            --data-impl mmap \\
            --split 949,50,1 \\
            --distributed-backend nccl \\
            --lr 0.0001 \\
            --lr-decay-style linear \\
            --min-lr 1.0e-5 \\
            --weight-decay 1e-2 \\
            --clip-grad 1.0 \\
            --lr-warmup-fraction .01 \\
            --log-interval 1 \\
            --save-interval 50 \\
            --eval-interval 50 \\
            --eval-iters 10 \\
            --tensor-model-parallel-size $TENSOR_MP_SIZE \\
            --pipeline-model-parallel-size $PIPELINE_MP_SIZE \\
            --DDP-impl {} \\
            --mpigraph_log_file {} \\
            --experiment_name {} \\
            --save_json_log \\
            --save_json_path {} \\
            --exp_id {} \\
            --sub_test_id {} \\
            --b2_jobid {} \\
            --gpu_type {} \\
            --p2p_dummy_size {} \\
            --assignment_json_file {} \\
            --opt_ranking_num {} \\
            --no-masked-softmax-fusion \\
            --no-bias-dropout-fusion \\
            --no-bias-gelu-fusion \\
""".format(num_layers, hidden_size, num_attention_heads, args.micro_batch_size, args.global_batch_size, seq_length, max_position_embeddings, args.train_iters, args.ddp_impl, MPIGRAPH_JSON_FILE, exp_name, JSON_LOG_PATH, args.exp_id, args.sub_test_id, args.b2_jobid,args.gpu_type,args.p2p_dummy_size,args.opt_output_file, args.opt_ranking_num)
    else:
        print('Unknown model type ...')
        exit()

# sc experiment arguments
    if args.fp_type == "fp16":
        lsf_string += \
"""
            --fp16 \\
"""
    elif args.fp_type == "df16":
        lsf_string += \
"""
            --bf16 \\
"""

# topo aware arguments 
    if args.topo_aware == 'True':
        lsf_string += \
"""
            --topo_aware \\
"""  
    lsf_string += "\""

    lsf_string += "\n"
    
    if args.network_speed == "200G" or args.network_speed == "100G":

        lsf_string += \
"""
NRANK=0
for HOST in $HOSTLIST
do
        if (( $NRANK == 0 )); then
                MASTER_ADDR=$HOST
                MCA_ARG="-report-bindings -mca pml_ucx_verbose 100 --bind-to none -x LD_LIBRARY_PATH -x PATH -mca pml ucx -x OMP_NUM_THREADS=64 -x NCCL_IB_HCA=mlx5_0:1 -x UCX_NET_DEVICES=mlx5_0:1"
                RUN_SCRIPT="-np 1 $MCA_ARG --host $MASTER_ADDR torchrun --nnodes=$NNODES --nproc_per_node=8 --max_restarts=0 --rdzv_id=111 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/{} $TOTAL_ARGS"
        else
                RUN_SCRIPT="$RUN_SCRIPT : -np 1 $MCA_ARG --host $HOST torchrun --nnodes=$NNODES --nproc_per_node=8 --max_restarts=0 --rdzv_id=111 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/{} $TOTAL_ARGS"
        fi
        NRANK=$(($NRANK+1))
done

echo "### RUN_SCRIPT RUN_SCRIPT RUN_SCRIPT ###"
echo $RUN_SCRIPT
echo ""

""".format(args.gpu_type, args.exp_alphabet, pre_run_script, args.gpu_type, args.exp_alphabet, pre_run_script)

    elif args.network_speed == "800G" :

        lsf_string += \
"""
NRANK=0
for HOST in $HOSTLIST
do
        if (( $NRANK == 0 )); then
                MASTER_ADDR=$HOST
                MCA_ARG="-report-bindings -mca pml_ucx_verbose 100 --bind-to none -x LD_LIBRARY_PATH -x PATH -mca pml ucx -x OMP_NUM_THREADS=64 -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1 -x UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1"
                RUN_SCRIPT="-np 1 $DEBUG $MCA_ARG $UCX_OPT --host $MASTER_ADDR --bind-to none -x PATH -x LD_LIBRARY_PATH torchrun --nnodes=$NNODES --nproc_per_node=8 --max_restarts=0 --rdzv_id=111 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/{} $TOTAL_ARGS"
        else
                RUN_SCRIPT="$RUN_SCRIPT : -np 1 $DEBUG $MCA_ARG $UCX_OPT --host $HOST --bind-to none -x PATH -x LD_LIBRARY_PATH torchrun --nnodes=$NNODES --nproc_per_node=8 --max_restarts=0 --rdzv_id=111 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/{} $TOTAL_ARGS"
        fi
        NRANK=$(($NRANK+1))
done

echo "### RUN_SCRIPT RUN_SCRIPT RUN_SCRIPT ###"
echo $RUN_SCRIPT
echo ""

""".format(args.gpu_type, args.exp_alphabet, pre_run_script, args.gpu_type, args.exp_alphabet, pre_run_script)
    else :
        print("Undefined network speed ...")
        exit()


    lsf_string += "\n"
    lsf_string += "python $PROFILE_GPU_MEM_PROGRAM --time_inter 1 --output_file {}profile_gpu_mem_{}.log &".format(PROFILE_LOG_PATH,exp_name)
    lsf_string += "\n"
    lsf_string += "mpirun $RUN_SCRIPT 2>&1 | tee ${{LOG_PATH}}/{}.log".format(exp_name)
    lsf_string += '\n'
    lsf_string += \
"""
mpirun -n $NNODES -x LD_LIBRARY_PATH -x PATH -hostfile ${CLEAN_GARBAGE_RANK_FILE} sh ${KILL_PROGRAM}
sleep 1
"""
    lsf_string += "\n"
    lsf_string += "python $MERGE_PROFILE_PROGRAM --read_gpu_file {}profile_gpu_mem_{}.log --read_train_file {}{}.json --write_file ${{MERGE_PROFILE_FILE}}".format(PROFILE_LOG_PATH, exp_name, JSON_LOG_PATH, exp_name)
    #f = open(str(args.exp_id) + "_" + exp_name +".sub" , 'w')
    #f = open(args.model_type + "_" +args.parameter_num + "_" + str(args.gpus) +"GPU_test" + str(args.exp_id)+"_s2" +".sub" , 'a')
    lsf_string += "\n"
    lsf_string += \
"""
#kill profile gpu mem program
ps -axf | grep python | grep jinkyu.yim | awk '{print $1}' | while read line;
do
    echo $line
    kill -9 $line
done
"""
    lsf_string += \
"""
sh /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5-{}-{}/clear.sh\n
""".format(args.gpu_type, args.exp_alphabet)
    lsf_string += \
"""
#check clear.sh end
ps -ef | grep -v grep | grep clear.sh
while [ $? == 0 ]
do
    sleep 1
    ps -ef | grep -v grep | grep clear.sh
done
"""
    lsf_string += \
"""
rm -rf ${CHECKPOINT_PATH}
"""
    lsf_string += "\n"
    f = open(args.run_job_file , 'a')
    f.write(lsf_string)
    f.close()

