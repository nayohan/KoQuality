# Works on RTX3090,RTX3090,A30 (VRam 24G) x4 + Ram 128GB / using deepspeed3 + cpu offloading
torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-1.3b' \
--train_file='KoAlpaca_v1.1a_textonly.json' \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=16 \
--torch_dtype=float16 \
--fp16 \
--output_dir='polyglot-1.3b-koalpaca-v1.1b' \
--deepspeed=ds_zero3_offload_fp16.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='polyglot-1.3b-koalpaca-v1.1b-ga16'