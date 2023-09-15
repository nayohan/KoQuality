# 1. make raw_corpus dataset
python prepare_raw_corpus.py 


# 2. calculate perplexity before clustering (PPL)
python generate_perplexity.py 


# 3. generate sentence embedding (EMB) and do length (L) based clustering (K)
python generate_sentence_embedding_clustering.py --len 1 --device 0 &
python generate_sentence_embedding_clustering.py --len 5 --device 1 &
python generate_sentence_embedding_clustering.py --len 10 --device 2 &
python generate_sentence_embedding_clustering.py --len 20 --device 3 &


# 4. sampling by method (M) with sampling ratio (N)
python proceed_sampling_dataset --len 1
python proceed_sampling_dataset --len 5
python proceed_sampling_dataset --len 10
python proceed_sampling_dataset --len 20