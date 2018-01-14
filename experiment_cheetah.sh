#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
#export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
python fwbw_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=4 --num_workers_trpo=32 --yaml_file='cheetah_forward' --fw_learning_rate 0.001 --bw_learning_rate 0.0005 --num_imagination_steps 50 --fw_iter 1 --top_k_trajectories 100 --policy_variance  0 --bw_model_hidden_size 64


#For running TRPO baseline run this
#python fwbw_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=4 --num_workers_trpo=32 --yaml_file='cheetah_forward' --running_baseline True
