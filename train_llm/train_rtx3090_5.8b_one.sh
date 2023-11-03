# Works on RTX3090,RTX3090,A30 (VRam 24G) x4 + Ram 256GB / using deepspeed3 + cpu offloading
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
# --model_name_or_path='hyunseoki/ko-ref-llama2-7b' \
# --train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/result_len5_k100_mppl_n0.01.json' \
# --num_train_epochs=5 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=32 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='KoQuality-ko-ref-llama2-7b' \
# --deepspeed=ds_zero3_offload_fp16.json \
# --do_train \
# --do_eval \
# --evaluation_strategy='steps' \
# --save_strategy='steps' \
# --logging_strategy='steps' \
# --save_steps=1 \
# --eval_steps=1 \
# --logging_steps=1 \
# --logging_first_step \
# --save_total_limit=1 \
# --load_best_model_at_end=True \
# --metric_for_best_model='accuracy' \
# --max_eval_samples=1000 \
# --run_name='KoQuality-ko-ref-llama2-7b'


# # koquality 0.01
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
--train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/result_len5_k100_mppl_n0.01.json' \
--num_train_epochs=5 \
--block_size=1024 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--torch_dtype=float16 \
--fp16 \
--output_dir='KoQuality-polyglot-ko-12.8b' \
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

# # koquality 0.01
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-1.3b' \
# --train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/result_len5_k100_mppl_n0.01.json' \
# --num_train_epochs=5 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=32 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='KoQuality-polyglot-ko-1.3b' \
# --deepspeed=ds_zero3_offload_fp16.json \
# --do_train \
# --do_eval \
# --evaluation_strategy='steps' \
# --save_strategy='steps' \
# --logging_strategy='steps' \
# --save_steps=1 \
# --eval_steps=1 \
# --logging_steps=1 \
# --logging_first_step \
# --save_total_limit=1 \
# --load_best_model_at_end=True \
# --metric_for_best_model='accuracy' \
# --max_eval_samples=1000 \
# --run_name='KoQuality-polyglot-ko-1.3b'


# poylglto-3.8b raw
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-3.8b' \
# --train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/koquality_raw.json' \
# --num_train_epochs=5 \
# --block_size=1024 \
# --per_device_train_batch_size=2 \
# --gradient_accumulation_steps=32 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='polyglot-3.8b-koquality_raw' \
# --deepspeed=ds_zero3_offload_fp16.json \
# --do_train \
# --do_eval \
# --evaluation_strategy='steps' \
# --save_strategy='steps' \
# --logging_strategy='steps' \
# --save_steps=10 \
# --eval_steps=10 \
# --logging_steps=10 \
# --logging_first_step \
# --save_total_limit=1 \
# --load_best_model_at_end=True \
# --metric_for_best_model='accuracy' \
# --max_eval_samples=1000 \
# --run_name='polyglot-3.8b-len10-koquality_raw'


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
# --model_name_or_path='hyunseoki/ko-ref-llama2-7b' \
# --train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/results_final_len10/len10_k100_mppl_n0.1.json' \
# --num_train_epochs=5 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=32 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='ko-ref-llama2-7b-len10_k100_mppl_n0.1' \
# --deepspeed=ds_zero3_offload_fp16.json \
# --do_train \
# --do_eval \
# --evaluation_strategy='steps' \
# --save_strategy='steps' \
# --logging_strategy='steps' \
# --save_steps=1 \
# --eval_steps=1 \
# --logging_steps=1 \
# --logging_first_step \
# --save_total_limit=1 \
# --load_best_model_at_end=True \
# --metric_for_best_model='accuracy' \
# --max_eval_samples=1000 \
# --run_name='ko-ref-llama2-7b-len10_k100_mppl_n0.1'

# beomi llama
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
# --model_name_or_path='beomi/llama-2-ko-7b' \
# --train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/results_final_len10/len10_k100_mppl_n0.1.json' \
# --num_train_epochs=5 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=32 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='llama-2-ko-7b-len10_k100_mppl_n0.1' \
# --deepspeed=ds_zero3_offload_fp16.json \
# --do_train \
# --do_eval \
# --evaluation_strategy='steps' \
# --save_strategy='steps' \
# --logging_strategy='steps' \
# --save_steps=1 \
# --eval_steps=1 \
# --logging_steps=1 \
# --logging_first_step \
# --save_total_limit=1 \
# --load_best_model_at_end=True \
# --metric_for_best_model='accuracy' \
# --max_eval_samples=1000 \
# --run_name='llama-2-ko-7b-len10-len10_k100_mppl_n0.1'

# Polyglot-ko koqaulity raw
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
# --train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/koquality_raw.json' \
# --num_train_epochs=5 \
# --block_size=1024 \
# --per_device_train_batch_size=2 \
# --gradient_accumulation_steps=32 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='polyglot-5.8b-koquality_raw' \
# --deepspeed=ds_zero3_offload_fp16.json \
# --do_train \
# --do_eval \
# --evaluation_strategy='steps' \
# --save_strategy='steps' \
# --logging_strategy='steps' \
# --save_steps=10 \
# --eval_steps=10 \
# --logging_steps=10 \
# --logging_first_step \
# --save_total_limit=1 \
# --load_best_model_at_end=True \
# --metric_for_best_model='accuracy' \
# --max_eval_samples=1000 \
# --run_name='polyglot-5.8b-len10-koquality_raw'

# # # Works on RTX3090,RTX3090,A30 (VRam 24G) x4 + Ram 256GB / using deepspeed3 + cpu offloading
# torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-1.3b' \
# --train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/result_len5_k100_mppl_n0.01.json' \
# --num_train_epochs=5 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=16 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='polyglot-1.3b-len10_k50_n001' \
# --deepspeed=ds_zero3-nooffload.json \
# --do_train \
# --do_eval \
# --evaluation_strategy='steps' \
# --save_strategy='steps' \
# --logging_strategy='steps' \
# --save_steps=1 \
# --eval_steps=1 \
# --logging_steps=1 \
# --logging_first_step \
# --save_total_limit=1 \
# --load_best_model_at_end=True \
# --metric_for_best_model='accuracy' \
# --max_eval_samples=1000 \
# --run_name='polyglot-1.3b-len10_k50_n001-b4-ga16'