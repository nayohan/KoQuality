# Works on A100 (VRam 80G) x4 + Ram 256GB / using deepspeed3 + no offloading

# len="1 5 10 20"
# kmeans="10 20 50 100"
# method="ppl_h ppl rand"
# smaple_n="0.1 0.05 0.01"

epoch=2
batch_size=4
ga=4

# len="1 5 10 20"
# kmeans="1"
# method="ppl_h ppl"
# smaple_n="0.01"

# len="10 20"
# kmeans="10 20 50 100"
# method="ppl_h ppl"
# smaple_n="0.01"

len="1 5 10 20 "
kmeans="1 10 20 50 100"
method="ppl ppl_h rand"
smaple_n="0.01"
for l in $len
do
    for k in $kmeans
    do
        for m in $method
        do
            for n in $smaple_n
            do
                train_path="train_dataset/final_results/"
                train_file="result_len${l}_k${k}_m${m}_n${n}"
                model_name="polyglot-ko-5.8b"

                torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
                --model_name_or_path="EleutherAI/${model_name}" \
                --train_file="${train_path}${train_file}.json" \
                --num_train_epochs=${epoch} \
                --block_size=1024 \
                --per_device_train_batch_size=${batch_size} \
                --gradient_accumulation_steps=${ga} \
                --torch_dtype=float16 \
                --fp16 \
                --output_dir="${model_name}-${train_file}_bs$(($batch_size * $ga))" \
                --deepspeed="ds_zero3-nooffload.json" \
                --do_train \
                --do_eval \
                --evaluation_strategy='steps' \
                --save_strategy='epoch' \
                --logging_strategy='steps' \
                --save_steps=1 \
                --eval_steps=1 \
                --logging_steps=1 \
                --logging_first_step \
                --save_total_limit=1 \
                --max_eval_samples=100 \
                --run_name="${train_file}_bs$(($batch_size * $ga))"
            done
        done
    done
done


# # Works on A100 80G x4
# l=20
# k=10
# m="ppl"
# n=0.01
# train_path="train_dataset/final_results_len${l}/"
# train_file="result_len${l}_k${k}_m${m}_n${n}"
# model_name="polyglot-ko-5.8b"

# torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
# --model_name_or_path="EleutherAI/${model_name}" \
# --train_file="${train_path}${train_file}.json" \
# --num_train_epochs=${epoch} \
# --block_size=1024 \
# --per_device_train_batch_size=${batch_size} \
# --gradient_accumulation_steps=${ga} \
# --learning_rate=5e-5 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir="${model_name}-${train_file}_bs$(($batch_size * $ga))" \
# --deepspeed="ds_zero3-nooffload.json" \
# --do_train \
# --do_eval \
# --evaluation_strategy='steps' \
# --save_strategy='epoch' \
# --logging_strategy='steps' \
# --save_steps=1 \
# --eval_steps=1 \
# --logging_steps=1 \
# --logging_first_step \
# --save_total_limit=1 \
# --max_eval_samples=100 \
# --run_name="${train_file}_bs$(($batch_size * $ga))"


# torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
# --train_file='KoAlpaca_v1.1a_textonly.json' \
# --num_train_epochs=1 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=32 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='polyglot-5.8b-koalpaca-v1.1b_steps_32' \
# --deepspeed=ds_flan_t5_z3_offload_bf16.json \
# --do_train \
# --save_strategy='epoch' \
# --logging_strategy='steps' \
# --logging_first_step \
# --save_total_limit=1 \
# --run_name='koalpaca-v1.1b_ga32'