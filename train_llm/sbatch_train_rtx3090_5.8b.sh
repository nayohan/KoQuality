#!/bin/sh

#SBATCH -o slurm-gpu-job.out
#SBATCH -e slurm-err.out
#SBATCH -p  normal
#SBATCH --gres=gpu:1

# --train_file='/home/closedai/.test/KoQuality/train_llm/train_dataset/result_len5_k100_mppl_n0.01.json' \

# # koquality 1
srun --gres=gpu:1 torchrun --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
--train_file='/home/closedai/.test/KoQuality/train_llm/train_dataset/koquality_raw.json' \
--num_train_epochs=5 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--torch_dtype=float16 \
--fp16 \
--output_dir='train_model/polyglot-ko-12.8b-inst' \
--deepspeed=ds_zero3_offload_fp16.json \
--do_train \
--do_eval \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--logging_strategy='steps' \
--save_steps=10 \
--eval_steps=10 \
--logging_steps=1 \
--logging_first_step \
--save_total_limit=1 \
--load_best_model_at_end=True \
--metric_for_best_model='accuracy' \
--max_eval_samples=1000 \
--run_name='polyglot-ko-12.8b-inst'

# # koquality 0.01
srun --gres=gpu:1 torchrun --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
--train_file='/home/closedai/.test/KoQuality/train_llm/train_dataset/result_len5_k100_mppl_n0.01.json' \
--num_train_epochs=5 \
--block_size=1024 \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=16 \
--torch_dtype=float16 \
--fp16 \
--output_dir='train_model/polyglot-ko-12.8b' \
--deepspeed=ds_zero3_offload_fp16.json \
--do_train \
--do_eval \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--logging_strategy='steps' \
--save_steps=1 \
--eval_steps=1 \
--logging_steps=1 \
--logging_first_step \
--save_total_limit=1 \
--load_best_model_at_end=True \
--metric_for_best_model='accuracy' \
--max_eval_samples=1000 \
--run_name='KoQuality-polyglot-ko-12.8b'
