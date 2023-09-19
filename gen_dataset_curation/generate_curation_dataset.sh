# 1. make raw_corpus dataset
python prepare_raw_corpus.py --save_hf_path nayohan/koquality_raw_test


# 2. calculate perplexity before clustering (PPL)
python generate_perplexity.py --device 3 -d nayohan/koquality_raw_test -m EleutherAI/polyglot-ko-1.3b --save_hf True --save_hf_user_name nayohan

# 3. generate sentence embedding (EMB) and do length (L) based clustering (K)
# python generate_sentence_embedding_clustering.py --len 1 --device 0 -d nayohan/koquality_raw_test -m EleutherAI/polyglot-ko-1.3b -e BM-K/KoSimCSE-roberta-multitask &
# python generate_sentence_embedding_clustering.py --len 5 --device 1 -d nayohan/koquality_raw_test -m EleutherAI/polyglot-ko-1.3b -e BM-K/KoSimCSE-roberta-multitask & 
python generate_sentence_embedding_clustering.py --len 10 --device 2 -d nayohan/koquality_raw_test -m EleutherAI/polyglot-ko-1.3b -e BM-K/KoSimCSE-roberta-multitask
# python generate_sentence_embedding_clustering.py --len 20 --device 3 -d nayohan/koquality_raw_test -m EleutherAI/polyglot-ko-1.3b -e BM-K/KoSimCSE-roberta-multitask &
wait

# 4. sampling by method (M) with sampling ratio (N)
# python proceed_sampling_dataset.py --len 1
# python proceed_sampling_dataset.py --len 5
python proceed_sampling_dataset.py --len 10
# python proceed_sampling_dataset.py --len 20