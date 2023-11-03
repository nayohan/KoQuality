# # Works on RTX3090,RTX3090,A30 (VRam 24G) x4 + Ram 256GB / using deepspeed3 + cpu offloading
torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-1.3b' \
--train_file='/home/uj-user/Yo/HiT5/HCLT/train_llm/train_dataset/results_final_len10/len10_k100_mrand_n0.1.json' \
--num_train_epochs=5 \
--block_size=1024 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--torch_dtype=float16 \
--fp16 \
--output_dir='polyglot-1.3b-len10_k100_n01' \
--deepspeed=ds_zero3-nooffload.json \
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
--run_name='polyglot-1.3b-len10_k100_n01-b4-ga16'


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