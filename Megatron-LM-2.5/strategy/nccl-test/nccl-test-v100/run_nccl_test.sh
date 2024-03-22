#!/bin/bash

#Program path
PROGRAM_NCCL_TEST="/home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/nccl-tests/build/sendrecv_perf"
MPI_MSG_SIZE_S=31457280
MPI_MSG_SIZE_E=31457280
GPU_PER_THREAD=1


FILE_PATH=$2
HOSTLIST=$1
HOSTS=`echo -e $HOSTLIST | sed ':a;N;$!ba;s/\n/_/g'`


MPI_RANK_FILE="${FILE_PATH}/mpi_rank_file_${HOSTS}"
NCCL_LOG_FILE="${FILE_PATH}/nccl_log_file_${HOSTS}"

HOST1=`echo $HOSTS | sed 's/_/ /g' | awk '{print $1}'`
HOST2=`echo $HOSTS | sed 's/_/ /g' | awk '{print $2}'`

echo $NCCL_LOG_FILE
echo $HOST1 $HOST2

if [ $HOST1 = $HOST2 ]; then
	
	MPI_SCRIPT="-x LD_LIBRARY_PATH -x PATH -H $HOST1 -mca pml ucx -mca btl ^openib -bind-to none -report-bindings" 

	NCCL_SCRIPT="$PROGRAM_NCCL_TEST -b $MPI_MSG_SIZE_S -e $MPI_MSG_SIZE_E -g 8"

	CMD="$MPI_SCRIPT $NCCL_SCRIPT"

else

	if [ -f $MPI_RANK_FILE ]; then
  		rm $MPI_RANK_FILE
	fi

	for HOST in $(echo -e $HOSTLIST)
	do
		echo "$HOST slots=4" >> $MPI_RANK_FILE
	done
	
	NNODES=`echo -e $HOSTLIST | wc -w`

	PROC_NUM=`expr $NNODES \* 4`


	MPI_SCRIPT="-x LD_LIBRARY_PATH -x PATH -hostfile $MPI_RANK_FILE -np $PROC_NUM -mca pml ucx -mca btl ^openib -bind-to none -report-bindings" 

	NCCL_SCRIPT="$PROGRAM_NCCL_TEST -b $MPI_MSG_SIZE_S -e $MPI_MSG_SIZE_E -g $GPU_PER_THREAD"

	CMD="$MPI_SCRIPT $NCCL_SCRIPT"

fi

mpirun $CMD 2>&1 | tee $NCCL_LOG_FILE

OUT_OF_PLACE=`cat $NCCL_LOG_FILE | grep -A2 busbw | tail -1 | sed -e 's/  */ /g' -e 's/^ *//g' -e 's/ *$//g' | awk '{print $8}'`
IN_PLACE=`cat $NCCL_LOG_FILE | grep -A2 busbw | tail -1 | sed -e 's/  */ /g' -e 's/^ *//g' -e 's/ *$//g' | awk '{print $12}'`
AVG_BW=`cat $NCCL_LOG_FILE | grep "Avg bus" | sed -e 's/  */ /g' -e 's/^ *//g' -e 's/ *$//g' | awk '{print $6}'`

echo $OUT_OF_PLACE $IN_PLACE $AVG_BW

FLOAT="^[+-]?[0-9]+\.?[0-9]*$"
if [[ $OUT_OF_PLACE =~ $FLOAT ]]
then
	echo $OUT_OF_PLACE
else
	echo "NCCL TEST FAILED, RETRY"
	mpirun $CMD
fi















