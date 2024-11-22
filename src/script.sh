#!/bin/bash
#SBATCH -J llama_finetune
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH --output=llama_finetune.out
#SBATCH --nodes=1
#SBATCH -c 30
#SBATCH --gres=gpu:3
#SBATCH --time=4-00:00:00

CUDA_VISIBLE_DEVICES=0,1,2 NCCL_DEBUG=INFO torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    finetune_fsdp.py
