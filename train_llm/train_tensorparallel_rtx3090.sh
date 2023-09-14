# Works on 4x RTX 3090/4090/A5000 (24G), using TensorParallel,

train_data_path="/home/uj-user/Yo/HiT5/HCLT/KoAlpaca/train_v1.1b/train_json/"
# train_file="KoAlpaca_v1.1a_textonly"
train_file_1="result_len5_k10_mppl_h_n0.1"
train_file_2="result_len5_k10_mppl_n0.1.json"
train_file_3="result_len5_k10_mrand_n0.1.json"

python run_tensor_parallel.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file="$train_data_path$train_file_1.json" \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=16 \
--fp16 \
--output_dir="polyglot-5.8b-$train_file_1" \
--do_train \
--optim='adafactor' \
--learning_rate='2e-5' \
--logging_strategy='steps' \
--logging_first_step \
--logging_steps=10 \
--run_name='polyglot-5.8b-koalpaca-v1.1a-A30-20epoch' \

python run_tensor_parallel.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file="$train_data_path$train_file_2.json" \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=16 \
--fp16 \
--output_dir="polyglot-5.8b-$train_file_2" \
--do_train \
--optim='adafactor' \
--learning_rate='2e-5' \
--logging_strategy='steps' \
--logging_first_step \
--logging_steps=10 \
--run_name='polyglot-5.8b-koalpaca-v1.1a-A30-20epoch' \
--remove_unused_columns false

python run_tensor_parallel.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file="$train_data_path$train_file_3.json" \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=16 \
--fp16 \
--output_dir="polyglot-5.8b-$train_file_3" \
--do_train \
--optim='adafactor' \
--learning_rate='2e-5' \
--logging_strategy='steps' \
--logging_first_step \
--logging_steps=10 \
--run_name='polyglot-5.8b-koalpaca-v1.1a-A30-20epoch' \
--low_cpu_mem_usage \
--remove_unused_columns false
--low_cpu_mem_usage \
--train_file='./KoAlpaca_v1.1a_textonly.json' \