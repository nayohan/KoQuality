# KoQuality
![image](https://github.com/nayohan/KoQuality/assets/18652811/93b46fbe-7d73-4ab6-aaf3-2ef47e889462)

<br/>

한국어 데이터셋에서 여러 명령어 튜닝 데이터셋의 통합하여 하나의 고품질 데이터셋을 만들고자 하였습니다. 기존의 데이터셋(KoAlpaca v1.1, Vicuna, Alpaca, Dolly, OIG)에서 명령어 문장을 큐레이션하여 고품질의 명령어 데이터셋인 KoQuality를 생성하였습니다. KoQuality는 기존 데이터셋의 1% (4.01K) 데이터셋으로 한국어 언어모델에 명령어 튜닝을 진행하였으며, 제로샷 KoBEST 벤치마크 및 Open-Ko-LLM 리더보드에서 기존의 모델과 비슷하거나 성능 향상을 보입니다.

<br/>

![image](https://github.com/nayohan/KoQuality/assets/18652811/dc822126-1a58-4d35-aba1-6271b66414ee)

(*2023-10-13 02:00기준, Polyglot5.8B 모델 비교) /
KoQuality 데이터셋은 [DILAB-HYU/KoQuality](https://huggingface.co/datasets/DILAB-HYU/KoQuality)에서  다운 받을 수 있습니다.

<br/>


## Sampled Examples
![image](https://github.com/nayohan/KoQuality/assets/18652811/f7d60417-29fb-42bc-b2dc-b61949f49fd1)

<br/>

## Results
KoQuality 데이터셋으로 학습된 [DILAB-HYU/KoQuality-Polyglot-5.8b](https://huggingface.co/DILAB-HYU/KoQuality-Polyglot-5.8b) 의 결과는 다음과 같습니다. 각 데이터셋 별로 큰 차이가 없는 경우나, Pertrained LLM보다는 높은성능을 보이지 못하는 결과를 보이기도 하지만, 전체적으로 성능향상 및 적은 데이터셋을 활용함으로써 오는 학습시간 단축의 효과가 있습니다.
<br/>

![image](https://github.com/nayohan/KoQuality/assets/18652811/82d6bb6f-f0f6-43ad-ba8d-ffaa4427350e)

<br/>


## Generate curation dataset (KoQuality)
고품질 명령어 데이터셋 KoQuality를 생성하는 방법은 다음과 같다. 
#### 0. 파이썬 환경 구축
```
python3.10 -m venv ko_venv
source ko_venv/bin/activate
pip install -r requirements.txt
```


#### 1. Prepare raw corpus 
먼저 Curation할 Dataset을 하나의 Raw corpus 데이터셋으로 만들어준다. 각각의 데이터셋마다 다른 Column_name을 Instruction, Output으로 병합하였다.
```
cd gen_dataset_curation
python prepare_raw_corpus.py --save_hf_path <username>/koquality_raw
```

#### 2. Calculate perplexity
Curation의 지표 중 하나인 Perplexity(PPL)계산을 위해 한국어 언어모델 [EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)모델을 활용하였다. 아래 코드는 PPL을 계산하고, 하나의 Column을 추가하여 데이터셋에 저장한다.

```
python generate_perplexity.py --device 3 -d <username>/koquality_raw -m EleutherAI/polyglot-ko-1.3b --save_hf True --save_hf_user_name <username>
```


#### 3. Generate sentence embeddings and proceed with clustering
Contrastive Learning으로 학습된 [BM-K/KoSimCSE-roberta](https://huggingface.co/BM-K/KoSimCSE-roberta-multitask)를 활용하여 명령어 문장을 Sentence Embedding하고, 이를 K-means Clustering으로 그룹화 하였다. 길이 그룹별 클러스터링을 진행하기 위해, 길이 그룹의 개수를 L : {1, 5, 10, 20}로, 클러스터링 개수를 K : {10, 20, 50, 100}으로 실험을 진행하였으며, 최종적으로 L=5, K=100을 활용한다.
```
python generate_sentence_embedding_clustering.py --len 1 --device 0 -d <username>/koquality_raw -m EleutherAI/polyglot-ko-1.3b -e BM-K/KoSimCSE-roberta-multitask &
python generate_sentence_embedding_clustering.py --len 5 --device 1 -d <username>/koquality_raw -m EleutherAI/polyglot-ko-1.3b -e BM-K/KoSimCSE-roberta-multitask & 
python generate_sentence_embedding_clustering.py --len 10 --device 2 -d <username>/koquality_raw -m EleutherAI/polyglot-ko-1.3b -e BM-K/KoSimCSE-roberta-multitask &
python generate_sentence_embedding_clustering.py --len 20 --device 3 -d <username>/koquality_raw -m EleutherAI/polyglot-ko-1.3b -e BM-K/KoSimCSE-roberta-multitask &
```

#### 4. Sampling dataset 
데이터셋을 Sampling하기 위해 샘플링 비율은 N : {0.1, 0.05, 0.01} 중 하나로 설정하여 데이터셋을 생성하였다. 생성된 데이터셋은 코드에 지정된 Folder에 저장된다.
```
python proceed_sampling_dataset.py --len 1
python proceed_sampling_dataset.py --len 5
python proceed_sampling_dataset.py --len 10
python proceed_sampling_dataset.py --len 20
```




## Train LLMs
생성한 데이터셋 KoQuality를 실험하기 위해 [Beomi/KoAlpaca](https://github.com/Beomi/KoAlpaca)에 작성된 코드를 활용하여 모델을 학습하였다. 학습에 활용한 모델은 [EleutherAI/polyglot-ko-5.8b](https://huggingface.co/EleutherAI/polyglot-ko-5.8b)를  A100 80GB 4장을 활용하여 학습하였으며, 실험 셋팅은 하단에 적어두었다.

```
cd train_llm
bash train_A100_5.8b.sh or train_rtx3090_5.8b.sh
```

```
learning_rate: 5e-5
train_batch_size: 4
seed: 42
distributed_type: multi-GPU (A100 80G)
num_devices: 4
gradient_accumulation_steps: 16
optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
lr_scheduler_type: linear
num_epochs: 2.0
```


## NLU Evaluation 
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)의 polyglot branch를 활용하여 KoBEST 벤치마크 데이텃세에 대해 평가를 진행하였다.
```

git clone https://github.com/EleutherAI/lm-evaluation-harness
git checkout polyglot

pip install -e .
pip install evaluate
pip install importlib_resource

bash run_ours.sh
```

## Citation
Please cite the repo if you use the data or code in this repo.
```
@article{2023koqaulity,
  title={KoQuality: 한국어 언어 모델을 위한 고품질 명령어 데이터 큐레이션},
  author={나요한 and 김다혜 and 채동규},
  journal={제 35 회 한글 및 한국어 정보처리 학술대회 (HCLT 2023)},
  pages={245--248},
  year={2023},
  publisher={한국정보과학회}
}
```
```
@misc{2023koqaulity,
  title = {KoQuality: Curation of High-quality Instruction Data for Korean Language Models},
  author = {Na, Yohan and Kim, Dahye and Chae, Dong-Kyu},
  journal={Proceedings of the 35th Annual Conference on Human and Cognitive Language Technology (HCLT 2023)},
  pages={},
  year = {2023},
}
```
## 
