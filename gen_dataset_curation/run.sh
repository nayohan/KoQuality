len_values=(1)
# k_values=(10 20 50 100)
# # m_values=('rand' 'ppl' 'ppl_h')
# # n_values=(0.05 1)

# length 별 클러스터링 생성
for len in "${len_values[@]}"
do
   python generate_sentence_embedding.py --len $len
done

# for len in "${len_values[@]}"
# do
#    for i in 0 1 2 3 # gpu id
#    do
#    echo $i 
#    echo ${k_values[i]}
#    # python main.py --len $len --k ${k_values[i]} --device $i
#    python main_bak.py --len $len --k ${k_values[i]}
#    done
#    wait
# done

python generate_sentence_embedding.py --len 1 --device 0 &
python generate_sentence_embedding.py --len 5 --device 1 &
python generate_sentence_embedding.py --len 10 --device 2 &
python generate_sentence_embedding.py --len 20 --device 3 &