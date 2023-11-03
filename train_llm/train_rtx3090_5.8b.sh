# Works on RTX3090,RTX3090,A30 (VRam 24G) x4 + Ram 128GB / using deepspeed3 + cpu offloading / 5.8B / batch 2
# Works on RTX3090,RTX3090,A30 (VRam 24G) x4 + Ram 384GB / using deepspeed3 + cpu offloading / 12.8B / batch 1

# len="1 5 10 20"
# kmeans="1 10 20 50 100"
# method="ppl_h ppl rand"
# smaple_n="0.1 0.05 0.01"

                # echo ${l}_${k}_${m}_${n}
                # echo 0.2/$n | bc
                # echo $train_file


# epoch=$(echo 0.2/$n | bc)
epoch=20
batch_size=1
ga="16"

# len="10"
# kmeans="10 20 50 100"
# method="ppl_h ppl rand"
# smaple_n="0.01"

len="5"
kmeans="100"
method="ppl"
smaple_n="0.01 0.05 0.1"
model_name="polyglot-ko-12.8b"

# for l in $len
# do
#     for k in $kmeans
#     do
#         for m in $method
#         do
#             for n in $smaple_n
#             do
#                 train_path="train_dataset/results_final_len${l}/"
#                 train_file="len${l}_k${k}_m${m}_n${n}"

#                 torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
#                 --model_name_or_path="EleutherAI/${model_name}" \
#                 --train_file="${train_path}${train_file}.json" \
#                 --num_train_epochs=${epoch} \
#                 --block_size=1024 \
#                 --per_device_train_batch_size=${batch_size} \
#                 --gradient_accumulation_steps=${ga} \
#                 --torch_dtype=float16 \
#                 --fp16 \
#                 --output_dir="${model_name}_${train_file}_bs$(($batch_size * $ga))" \
#                 --deepspeed="ds_zero3_offload_fp16.json" \
#                 --do_train \
#                 --do_eval \
#                 --max_steps=200 \
#                 --evaluation_strategy='steps' \
#                 --save_strategy='steps' \
#                 --logging_strategy='steps' \
#                 --save_steps=10 \
#                 --eval_steps=10 \
#                 --logging_steps=1 \
#                 --logging_first_step \
#                 --save_total_limit=1 \
#                 --load_best_model_at_end=True \
#                 --metric_for_best_model='accuracy' \
#                 --run_name="${train_file}_bs$(($batch_size*$ga))"
#             done
#         done
#     done
# done

# old-ppl
for l in $len
do
    for k in $kmeans
    do
        for m in $method
        do
            for n in $smaple_n
            do
                train_path="old_train_dataset/results_final_len${l}/"
                train_file="result_len${l}_k${k}_m${m}_n${n}"

                torchrun --nproc_per_node=2 --master_port=34321 run_clm.py \
                --model_name_or_path="EleutherAI/${model_name}" \
                --train_file="${train_path}${train_file}.json" \
                --num_train_epochs=${epoch} \
                --block_size=1024 \
                --per_device_train_batch_size=${batch_size} \
                --gradient_accumulation_steps=${ga} \
                --torch_dtype=float16 \
                --fp16 \
                --output_dir="oldppl-${model_name}_${train_file}_bs$(($batch_size * $ga))" \
                --deepspeed="ds_zero3_offload_fp16.json" \
                --do_train \
                --do_eval \
                --max_steps=200 \
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
                --run_name="oldppl-${train_file}_bs$(($batch_size*$ga))"
            done
        done
    done
done

## other dataset train (reproduce koalpaca)
# epoch=5
# train_dataset="KoAlpaca_v1.1a_textonly"
# for train_file in $train_dataset
# do
#     train_path="train_dataset/"
#     torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
#     --model_name_or_path="EleutherAI/${model_name}" \
#     --train_file="${train_path}${train_file}.json" \
#     --num_train_epochs=${epoch} \
#     --block_size=1024 \
#     --per_device_train_batch_size=${batch_size} \
#     --gradient_accumulation_steps=${ga} \
#     --torch_dtype=float16 \
#     --fp16 \
#     --output_dir="${model_name}_${train_file}_bs$(($batch_size * $ga))" \
#     --deepspeed="ds_zero3_offload_fp16.json" \
#     --do_train \
#     --do_eval \
#     --max_steps=200 \
#     --evaluation_strategy='steps' \
#     --save_strategy='steps' \
#     --logging_strategy='steps' \
#     --save_steps=10 \
#     --eval_steps=10 \
#     --logging_steps=1 \
#     --logging_first_step \
#     --save_total_limit=1 \
#     --load_best_model_at_end=True \
#     --metric_for_best_model='accuracy' \
#     --run_name="${train_file}_bs$(($batch_size*$ga))"
# done

# # Run once using koquality
# torchrun --nproc_per_node=4 --master_port=34321 run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
# --train_file='train_json/result_len5_k100_mppl_n0.1.json' \
# --num_train_epochs=1 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=32 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='polyglot-5.8b-result_len5_k100_mppl_n0.1_steps_32' \
# --deepspeed=ds_flan_t5_z3_offload_bf16.json \
# --do_train \
# --save_strategy='epoch' \
# --logging_strategy='steps' \
# --logging_first_step \
# --save_total_limit=1 \
# --run_name='len5_k10_mppl_n0.1_ga32'

# # Run once using koalapca
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