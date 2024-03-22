#!/bin/bash

PROGRAM_NCCL_SCRIPT="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/nccl-test/nccl-test-a100/run_nccl_test.sh"
PROGRAM_ROUND_ROBIN="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/strategy/nccl-test/nccl-test-a100/round_robin_test.py"

HOSTFILE=$1
DATAPATH=$2
OUTPUTFILE=$3



python $PROGRAM_ROUND_ROBIN --run_script $PROGRAM_NCCL_SCRIPT --data_path "$DATAPATH" --hostlist $HOSTFILE --output_file "$OUTPUTFILE"

