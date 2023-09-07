len_values=(1 3 5)
# k_values=(10 20 50 100)
# # m_values=('rand' 'ppl' 'ppl_h')
# # n_values=(0.05 1)

# # 모든 조합에 대한 반복
for len in "${len_values[@]}"
do
   python main_bak.py --len $len
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