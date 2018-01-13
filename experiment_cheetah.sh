#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
#export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
python fwbw_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=4 --num_workers_trpo=32 --yaml_file='cheetah_forward'
#For running TRPO baseline run this
#python fwbw_mf.py --seed=0 --save_trpo_run_num=1 --which_agent=4 --num_workers_trpo=32 --yaml_file='cheetah_forward' --running_baseline True
