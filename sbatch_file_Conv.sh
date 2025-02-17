#!/bin/bash

#SBATCH --job-name=gpn-conv
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1 
#SBATCH --nodelist=ailb-login-03      
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0 
#SBATCH --partition=all_serial    
#SBATCH --account=ai4bio2023
#SBATCH --error=slurm_learning/log/gpn-Conv%A_%a.err           
#SBATCH --output=slurm_learning/log/gpn-Conv%A_%a.out         

module unload cuda/12.1
module load cuda/11.8
. /usr/local/anaconda3/etc/profile.d/conda.sh

export PYTHONPATH="${PYTHONPATH}:/homes/fvincenzi/firstenv/bin/python"

srun python -u gpn/ss/run_mlm.py --do_train=True --do_test=True --do_eval=True --report_to="wandb" \
 --prediction_loss_only=False --remove_unused_columns=False --dataset_name="results/dataset" \
 --tokenizer_name="gonzalobenegas/tokenizer-dna-mlm" --soft_masked_loss_weight_train=0.1 \
 --soft_masked_loss_weight_evaluation=0.0 --weight_decay=0.01 --optim="adamw_torch" \
 --dataloader_num_workers=2 --seed=42 --save_strategy="steps" --save_steps=1000 --evaluation_strategy="steps" --eval_steps=1000 --logging_steps=500 --max_steps=25000 --warmup_steps=100 --learning_rate=1e-3 \
 --lr_scheduler_type="constant_with_warmup" --run_name="gpn_run_Conv_example", --output_dir="/work/ai4bio2023/genomic_train_job2" --model_type="ConvNet" --per_device_train_batch_size=16 --per_device_eval_batch_size=16 --gradient_accumulation_steps=1 --config_overrides=n_layers=20 --max_train_samples=50000 --max_eval_samples=10000 --max_test_samples=10000 --ddp_find_unused_parameters=False --overwrite_output_dir