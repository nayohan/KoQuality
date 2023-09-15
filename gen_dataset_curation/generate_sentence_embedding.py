import os

import tqdm
import pandas as pd
import numpy as np
import torch
# import datasets
import copy
import argparse
import glob
from glob import glob
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModel

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from datasets import Dataset, DatasetDict


class DataSelect:
    def __init__(self, df):
        self.df = df

    def make_split_len(self, len):
        df_list = [] # df 다섯개 담을 리스트 선언
        for i in range(int(len)):
            if i == len-1:
                now_len_df = self.df[self.df['len'] >= self.df['len'].quantile(q=((i)/len))]
            else:
                now_len_df = self.df[(self.df['len'] >= self.df['len'].quantile(q=((i)/len))) & (self.df['len'] < self.df['len'].quantile(q=((i+1)/len)))]
            now_len_df['len_group'] = i
            df_list.append(now_len_df)
        return df_list

    def make_embedding(self, df_list, tokenizer, model, device):
        np_emb_all_list = []
        for df in df_list:
            emb_all = []
            for sent in tqdm(df['instruction']):
                max_sequence_length = 512
                token = tokenizer(sent, padding="max_length", truncation=True, max_length=max_sequence_length, return_tensors="pt")
                outputs = model(**token.to(device))
                sent_pooler_output = outputs.pooler_output
                emb_all.append(sent_pooler_output.detach().cpu().numpy().squeeze())

            np_emb_all = np.array(emb_all)
            np_emb_all_list.append(np_emb_all)
        return np_emb_all_list

    def run_clustering(self, len, df_list, np_emb_all_list, K):
        temp = copy.deepcopy(df_list)
        for i in range(len):
            model = KMeans(random_state=42, n_clusters=K, n_init=10)
            model.fit(np_emb_all_list[i])
            text_clusterindex = model.fit_predict(np_emb_all_list[i])
            temp[i]['cluster'] = text_clusterindex
        return temp


def concat_df_list(new_df_list): # length 별 쪼개진 5개 concat
    final_df = pd.concat(new_df_list, ignore_index=True)
    return final_df

def sampling_data(df_list, K, M, N):
    new_df_list = []
    for df in df_list:
        if M == 'rand':
            df = df.sample(frac=N)
        elif M == 'ppl':
            df = df.sort_values('ppl')
            df_cluster_list = [df['cluster']==k for k in range(K)]
            clustered_df_list = [df[df_bool] for df_bool in df_cluster_list]
            
            clustered_ppl_df_list = []
            for clustered_df in clustered_df_list:
                small_range = list(range(0, len(clustered_df), int(1 / N)))
                clustered_ppl_df = clustered_df.iloc[small_range, :]
                clustered_ppl_df_list.append(clustered_ppl_df)
            df = pd.concat(clustered_ppl_df_list, ignore_index=True)
            
        elif M == 'ppl_h':
            df = df.sort_values('ppl', ascending=False)
            df_cluster_list = [df['cluster']==k for k in range(K)]
            clustered_df_list = [df[df_bool] for df_bool in df_cluster_list]
            
            clustered_ppl_df_list = []
            for clustered_df in clustered_df_list:
                prop = int(N * len(clustered_df))
                clustered_df = clustered_df.drop_duplicates(['ppl'])
                print('c:', clustered_df)
                clustered_ppl_df = clustered_df[:prop]
                clustered_ppl_df_list.append(clustered_ppl_df)
            df = pd.concat(clustered_ppl_df_list, ignore_index=True)

        new_df_list.append(df)
    return new_df_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data using DataSelect class")
    parser.add_argument("--len", type=int, help="make_split_len")
    parser.add_argument("--k", type=int, help="Number of clusters for task 'run_clustering'")
    # parser.add_argument("--m", type=str, help="Sampling method for task 'sampling_data'")
    # parser.add_argument("--n", type=float, help="Sampling ratio for task 'sampling_data'")
    parser.add_argument("--device", type=int, default=0, help="Gpu device number")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # len_values = [1, 5]
    k_values = [1, 10, 20, 50, 100]
    m_values = ['rand']#, 'ppl_h']
    n_values = [0.01]#, 0.05, 0.1]


    # 2. Perplexity 계산하여 업데이트
    # use gen_perplexity
    
    # 3. 임베딩 생성 및 길이기반 클러스터링
    # ppl_dataset_path = '/home/uj-user/Yo/HiT5/HCLT/gen_perplexity/result_ppl/'
    # ppl_df = pd.read_json(f'{ppl_dataset_path}koquality_raw_ppl_polyglot-ko-1.3b_nospace.json', lines=True, orient='records')
    # # ppl_df = ppl_df.loc[:10000, :]
    # ppl_df['len_group']=""
    # calc = DataSelect(ppl_df)
    
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask').to(device)
    # tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')
    
    # split_len_df_list = calc.make_split_len(args.len)
    # np_emb_all_list = calc.make_embedding(calc.make_split_len(args.len), tokenizer, model, device)
    
    # for k_val in k_values:
    #     clustered_df_list = calc.run_clustering(args.len, split_len_df_list, np_emb_all_list, k_val)
    #     final_df = pd.concat(clustered_df_list, ignore_index=True)
    #     final_df = final_df.sort_values(by=['len_group', 'cluster', 'ppl', 'len'], axis=0)
    #     final_df = final_df[["ppl", "len", "len_group", "cluster", "group", "instruction", "output"]]

    #     output_dir = f"result_ppl_instruction_len{args.len}"
    #     os.makedirs(output_dir, exist_ok=True)
    #     filename = f"result_len{args.len}_k{k_val}.json"
    #     output_path = os.path.join(output_dir, filename)
    #     final_df.to_json(output_path, orient='records', lines=True, force_ascii=False)

        # upload to huggingface dataset
        # raw_train = Dataset.from_pandas(final_df)
        # final_dataset = DatasetDict({'train': raw_train})
        # final_dataset = final_dataset.remove_columns(['ppl  '])
        # final_dataset.push_to_hub(f'nayohan/koquality_len{args.len}_k{k_val}')
        
        
    # 4. Method 샘플링 및 샘플링 N%
    ppl_dataset_path = f'/home/uj-user/Yo/HiT5/HCLT/gen_dataset_curation/result_ppl_instruction_len{args.len}'
    ppl_dataset_list = glob(ppl_dataset_path+'/*')
    # print(ppl_dataset_list)
    for k_val in k_values:
        # print(f'{ppl_dataset_path}/result_len1_k{k_val}.json')
        ppl_df = pd.read_json(f'{ppl_dataset_path}/result_len{args.len}_k{k_val}.json', lines=True, orient='records')
        # print('ppl_df',ppl_df)
        ppl_df_bool_list = [ppl_df['len_group']==len for len in range(args.len)]
        # print(ppl_df_bool_list)
        clustered_df_list = [ppl_df[df_bool] for df_bool in ppl_df_bool_list]
        # print(clustered_df_list)
        
        for m_val in m_values:
            for n_val in n_values:
                sampled_df_list = sampling_data(clustered_df_list, k_val, m_val, n_val)
                result = concat_df_list(sampled_df_list)
                result = result.sort_values(by=['len_group', 'cluster', 'ppl', 'len'], axis=0)
                # 결과를 저장할 디렉토리 생성
                output_dir = f"final_results"
                os.makedirs(output_dir, exist_ok=True)

                # 파일 이름 생성
                filename = f"result_len{args.len}_k{k_val}_m{m_val}_n{n_val}.json"
                output_path = os.path.join(output_dir, filename)

                # 데이터프레임을 JSON 파일로 저장
                # final_df = final_df[["ppl", "len", "len_group", "cluster", "group", "instruction", "output"]]
                # final_result = pd.DataFrame({"cluster": result['cluster'], "ppl": result['ppl'], "instruction": result['instruction'], "output": result['output']})
                result.to_json(output_path, orient='records', lines=True, force_ascii=False)