#!/bin/bash


ITERS=10
WINDOW=10
#Program path
MPI_PROGRAM_PATH="../mpigraph"
PROGRAM_SERVER_MPIGRAPH="${MPI_PROGRAM_PATH}/mpiGraph"
PROGRAM_SERVER_NUMARUN_MPIGRAPH="${MPI_PROGRAM_PATH}/numa_run_mpigraph_200g.sh"
PROGRAM_MPIGRAPH_LOG_PARSER="${MPI_PROGRAM_PATH}/mpiGraph_log_parser.sh"
PROGRAM_MPIGRAPH_LOG_TO_CSV="${MPI_PROGRAM_PATH}/log_to_csv_200G.py"
PROGRAM_VIVALDI_CSV_TO_XY="$PROGRAM_PATH/vivaldi-master/read_mpigraph.py"
PROGRAM_MPIGRAPH_CRUNCH="/home/sr5/jinkyu.yim/project/ai_cluster_check/3_net_mpigraph/crunch_mpiGraph"

MPI_MSG_SIZE=$1
HOSTLIST=$2
MPIGRAPH_RANK_FILE=$3
MPIGRAPH_LOG_PATH=$4

echo "mpigraph go $MPI_MSG_SIZE"
echo "mpigraph go $HOSTLIST"
echo "mpigraph_rank_file $MPIGRAPH_RANK_FILE"
echo "mpigraph_log_path $MPIGRAPH_LOG_PATH"


MPIGRAPH_LOG_FILE="${MPIGRAPH_LOG_PATH}/mpigraph_raw_log_file"
MPIGRAPH_LOG_SEND_FILE="${MPIGRAPH_LOG_PATH}/mpigraph_send_file.log"
MPIGRAPH_LOG_RECV_FILE="${MPIGRAPH_LOG_PATH}/mpigraph_recv_file.log"
MPIGRAPH_LOG_JSON_FILE="${MPIGRAPH_LOG_PATH}/mpigraph_file.json"
MPIGRAPH_LOG_VIVALDI_FILE="${MPIGRAPH_LOG_PATH}/mpigraph_vivaldi_xy.log"


HOSTLIST=`cat $HOSTLIST`
NNODES=`echo $HOSTLIST | wc -w`

echo "mpigraph go $MPI_MSG_SIZE"
echo "mpigraph go $HOSTLIST"
echo "mpigraph_rank_file $MPIGRAPH_RANK_FILE"
echo "mpigraph_log_path $MPIGRAPH_LOG_PATH"

if [ ! -f "$MPIGRAPH_RANK_FILE" ]; then
	for HOST in $HOSTLIST
	do
		echo "$HOST slots=1" >> $MPIGRAPH_RANK_FILE
	done
fi

PROC_NUM=`expr $NNODES \* 1`

echo $PROC_NUM
RUN_SCRIPT="-n $PROC_NUM -x LD_LIBRARY_PATH -x PATH -hostfile $MPIGRAPH_RANK_FILE -report-bindings -mca btl ^openib -mca pml ucx --bind-to none $PROGRAM_SERVER_NUMARUN_MPIGRAPH --bin $PROGRAM_SERVER_MPIGRAPH $MPI_MSG_SIZE $ITERS $WINDOW"

mpirun $RUN_SCRIPT 2>&1 | tee $MPIGRAPH_LOG_FILE 

$PROGRAM_MPIGRAPH_CRUNCH $MPIGRAPH_LOG_FILE

sh $PROGRAM_MPIGRAPH_LOG_PARSER $MPIGRAPH_LOG_FILE $MPIGRAPH_LOG_SEND_FILE $MPIGRAPH_LOG_RECV_FILE


python $PROGRAM_MPIGRAPH_LOG_TO_CSV --recv_file $MPIGRAPH_LOG_RECV_FILE --send_file $MPIGRAPH_LOG_SEND_FILE --write_file $MPIGRAPH_LOG_JSON_FILE --node_size $NNODES


 
