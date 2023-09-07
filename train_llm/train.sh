# Works on A100 80G x4
torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='train_json/result_len5_k10_mppl_h_n0.1.json' \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--torch_dtype=float16 \
--fp16 \
--output_dir='polyglot-5.8b-result_len5_k10_mppl_h_n0.1_steps_32' \
--deepspeed=ds_flan_t5_z3_offload_bf16.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='len5_k10_mppl_h_n0.1_ga32'


torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='train_json/result_len5_k10_mppl_n0.1.json' \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--torch_dtype=float16 \
--fp16 \
--output_dir='polyglot-5.8b-result_len5_k10_mppl_n0.1_steps_32' \
--deepspeed=ds_flan_t5_z3_offload_bf16.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='len5_k10_mppl_n0.1_ga32'



torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='train_json/result_len5_k10_mrand_n0.1.json' \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--torch_dtype=float16 \
--fp16 \
--output_dir='polyglot-5.8b-result_len5_k10_mrand_n0.1_steps_32' \
--deepspeed=ds_flan_t5_z3_offload_bf16.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='len5_k10_mrand_n0.1_ga32'


torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='KoAlpaca_v1.1a_textonly.json' \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--torch_dtype=float16 \
--fp16 \
--output_dir='polyglot-5.8b-koalpaca-v1.1b_steps_32' \
--deepspeed=ds_flan_t5_z3_offload_bf16.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='koalpaca-v1.1b_ga32'