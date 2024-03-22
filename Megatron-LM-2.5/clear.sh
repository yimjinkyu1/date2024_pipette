#!/bin/bash


rm -rf /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5/megatron/fused_kernels/build

find /home/sr5/jinkyu.yim/ai_framework/Megatron-LM-2.5 | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
