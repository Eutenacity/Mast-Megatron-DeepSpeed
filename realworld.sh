CUDA_HOME="/PATH/TO/CUDA"
CODE_PATH="/PATH/TO/CODE"
PYTHON_PATH="/PATH/TO/PYTHON"
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:/home/wenxianglin/openmpi/lib
MASTER_ADD=ibgpu2
run_cmd=" LD_LIBRARY_PATH=$LD_LIBRARY_PATH CUDA_HOME=$CUDA_HOME bash examples_deepspeed/MoE/ds_pretrain_gpt_1.3B_MoE128.sh"

eval_code="${run_cmd} 0 $PYTHON_PATH $CODE_PATH $MASTER_ADD"
ssh -n -f ibgpu5 "sh -c 'cd $CODE_PATH; ${run_cmd} 1 $PYTHON_PATH $CODE_PATH $MASTER_ADD'"&
# ssh -n -f ibgpu3 "sh -c 'cd $CODE_PATH; ${run_cmd} 2 $PYTHON_PATH $CODE_PATH $MASTER_ADD'"&
# ssh -n -f ibgpu4 "sh -c 'cd $CODE_PATH; ${run_cmd} 3 $PYTHON_PATH $CODE_PATH $MASTER_ADD'"&
# ssh -n -f ibgpu5 "sh -c 'cd $CODE_PATH; ${run_cmd} 4 $PYTHON_PATH $CODE_PATH $MASTER_ADD'"&
# ssh -n -f ibgpu6 "sh -c 'cd $CODE_PATH; ${run_cmd} 5 $PYTHON_PATH $CODE_PATH $MASTER_ADD'"&
# ssh -n -f ibgpu8 "sh -c 'cd $CODE_PATH; ${run_cmd} 6 $PYTHON_PATH $CODE_PATH $MASTER_ADD'"&
# ssh -n -f ibgpu5 "sh -c 'cd $CODE_PATH; ${run_cmd} 7 $PYTHON_PATH $CODE_PATH $MASTER_ADD'"&

eval ${eval_code}
