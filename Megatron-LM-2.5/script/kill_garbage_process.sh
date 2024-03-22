#!/bin/bash

GPU_PID=""
#GPU_PID=`ps -aux | grep pretrain_gpt | grep -v grep | awk '{print $2}'`
GPU_PID=`ps -aux | grep pretrain_gpt | grep -v grep`
if [ "$GPU_PID" == "" ];then
	echo "$HOSTNAME empty ok"
	echo $GPU_PID
else
	echo "$HOSTNAME zombie process"
	echo $GPU_PID
	kill -9 $(ps -aux | grep pretrain_gpt | grep -v grep | awk '{print $2}')
fi

GPU_MEM_PID=`ps -axf | grep run_profile_gpumem | grep -v grep`
if [ "$GPU_MEM_PID" == "" ];then
	echo "$HOSTNAME gpu mem empty ok"
	echo $GPU_MEM_PID
else
	echo "$HOSTNAME gpu mem process"
	echo $GPU_MEM_PID
	kill -9 $(ps -axf | grep run_profile_gpumem | grep -v grep | awk '{print $1}')
fi

#kill -9 $(ps -aux | grep pretrain_gpt | grep -v grep | awk '{print $2}')
#rm temp.log
#TEXT=`ps -aux | grep pretrain_gpt | grep -v grep | awk '{print $2}'`
#echo "$TEXT" >> temp.log
