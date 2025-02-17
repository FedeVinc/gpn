#!/bin/bash

module unload cuda/12.1
module load cuda/11.8
. /usr/local/anaconda3/etc/profile.d/conda.sh

srun -Q --immediate=30 --mem=8G --partition=all_serial --account=ai4bio2023 --gres=gpu:1 --time 1:00:00 --pty -w ailb-login-03 bash