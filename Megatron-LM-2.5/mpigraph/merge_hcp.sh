#! /bin/bash

#https://github.com/LLNL/mpiGraph

MSG_SIZE=$1 #messages of size bytes 
HOSTFILE=$2 #nodes 

DATE=$(date +%Y%m%d_%H-%M-%S) 

PROGRAM_PATH="/home/sr6/jinkyu.yim/Megatron-CC-main/merge"
PROGRAM_SERVER_NUMARUN_MPIGRAPH="$PROGRAM_PATH/mpigraph/numa_run_mpigraph.sh"
PROGRAM_SERVER_MPIGRAPH="$PROGRAM_PATH/mpigraph/mpiGraph"
PROGRAM_MPIGRAPH_LOG_PARSER="$PROGRAM_PATH/mpigraph/mpiGraph_log_parser.sh"
PROGRAM_MPIGRAPH_LOG_TO_CSV="$PROGRAM_PATH/mpigraph/log_to_csv.py"
PROGRAM_VIVALDI_CSV_TO_XY="$PROGRAM_PATH/vivaldi-master/read_mpigraph.py"

MPIGRAPH_RANK_FILE="$PROGRAM_PATH/mpigraph/mpigraph_rank_file_$DATE"
MPIGRAPH_LOG_RAW_FILE="$PROGRAM_PATH/mpigraph/mpigraph_log_raw_file_$DATE.log"
MPIGRAPH_LOG_SEND_FILE="$PROGRAM_PATH/mpigraph/mpigraph_log_send_file_$DATE.log"
MPIGRAPH_LOG_RECV_FILE="$PROGRAM_PATH/mpigraph/mpigraph_log_recv_file_$DATE.log"
MPIGRAPH_LOG_JSON_FILE="$PROGRAM_PATH/mpigraph/mpigraph_log_json_file_$DATE.log"

HOSTLIST=`cat $HOSTFILE`
NNODES=`echo $HOSTLIST | wc -w`

mpigraph_rank_options() {
	for HOST in $HOSTLIST
	do
		echo "$HOST slots=4" >> $MPIGRAPH_RANK_FILE
	done
}

mpi_mpigraph_run_options() {
	PROC_NUM=`expr $NNODES \* 4`
	RUN_SCRIPT="-n $PROC_NUM -x LD_LIBRARY_PATH -x PATH -hostfile $MPIGRAPH_RANK_FILE -report-bindings -mca btl ^openib -mca pml ucx --bind-to none $PROGRAM_SERVER_NUMARUN_MPIGRAPH --bin $PROGRAM_SERVER_MPIGRAPH $MSG_SIZE $ITERS $WINDOW"
}

mpigraph_rank_options
ITERS=10
WINDOW=10
mpi_mpigraph_run_options
mpirun $RUN_SCRIPT 2>&1 | tee $MPIGRAPH_LOG_RAW_FILE

#sleep 5

sh $PROGRAM_MPIGRAPH_LOG_PARSER $MPIGRAPH_LOG_RAW_FILE $MPIGRAPH_LOG_SEND_FILE $MPIGRAPH_LOG_RECV_FILE

python $PROGRAM_MPIGRAPH_LOG_TO_CSV --recv_file $MPIGRAPH_LOG_RECV_FILE --send_file $MPIGRAPH_LOG_SEND_FILE --write_file $MPIGRAPH_LOG_JSON_FILE --node_size $NNODES

python $PROGRAM_VIVALDI_CSV_TO_XY $MPIGRAPH_LOG_JSON_FILE
#
#mpigraph csv file to json (each node's bandwidth)
#/home/sr5/jbeen.lee/mpigraph_log/20230413_mpi_result/make_dataframe.ipynb

#run mpigraph
#txt to csv 

#csv to json 

#xy coordinate 


#transform xy coordinate
#/home/sr6/jinkyu.yim/Megatron-CC-main/network_coordinate/vivaldi-master/read_mpigraph.py


#clustering 
#/home/sr5/yr9.choi/clustering/16server_network_plot.py 


#pipelining 



 
