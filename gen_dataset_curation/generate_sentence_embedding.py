import os

import tqdm
import pandas as pd
import numpy as np
import torch
# import datasets
import copy
import argparse
import glob
import preprocess_dataset
from glob import glob
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModel

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans


class DataSelect:
    def __init__(self, df):
        self.df = df

    def make_split_len(self, len):
        df_list = [] # df 다섯개 담을 리스트 선언
        for i in range(len):
            if i == len-1:
                now_len_df = self.df[self.df['len'] >= self.df['len'].quantile(q=((i)/len))]
            else:
                now_len_df = self.df[(self.df['len'] >= self.df['len'].quantile(q=((i)/len))) & (self.df['len'] < self.df['len'].quantile(q=((i+1)/len)))]
            # now_len_df['len_group'] = i
            now_len_df.loc[:, 'len_group'] = i
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
            model = KMeans(random_state=42, n_init=10)
            visualizer = KElbowVisualizer(model, k=(1, K))
            visualizer.fit(np_emb_all_list[i])
            text_clusterindex = model.fit_predict(np_emb_all_list[i])
            temp[i]['cluster'] = text_clusterindex
        return temp

    def sampling_data(self, df_list, M, N):
        new_df_list = []
        for df in df_list:
            if M == 'rand':
                df = df.sample(frac=N)
            elif M == 'ppl':
                df = df.sort_values('ppl')
                small_range = list(range(0, len(df), int(1 / N)))
                df = df.iloc[small_range, :]
            elif M == 'ppl_h':
                prop = int(N * len(df))
                df = df.sort_values('ppl', ascending=False)
                df = df[:prop]
            new_df_list.append(df)
        return new_df_list

    def concat_df_list(self, new_df_list): # length 별 쪼개진 5개 concat
        final_df = pd.concat(new_df_list, ignore_index=True)
        return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data using DataSelect class")
    parser.add_argument("--len", type=int, help="make_split_len")
    parser.add_argument("--k", type=int, help="Number of clusters for task 'run_clustering'")
    # parser.add_argument("--m", type=str, help="Sampling method for task 'sampling_data'")
    # parser.add_argument("--n", type=float, help="Sampling ratio for task 'sampling_data'")
    parser.add_argument("--device", type=int, help="Gpu device number")
    args = parser.parse_args()
    
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" #str(args.device)

    print('device:', args.device)
    # 여러 인자 값을 리스트로 정의합니다.
    # len_values = [1, 5]
    k_values = [50, 100]
    m_values = ['rand', 'ppl', 'ppl_h']
    n_values = [0.01, 0.05, 0.1]

    # 모든 조합에 대한 반복
    # for len_val in args.len:
         # 파서를 사용하여 인자를 설정합니다.

    # 모델 로드
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask').to(device)
    tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')

    # DataSelect 인스턴스 생성
    df = preprocess_dataset.make_dataset()
    
    # from datasets import Dataset, DatasetDict
    # raw_train = Dataset.from_pandas(df)
    # concat_dataset = DatasetDict({'train': raw_train})
    # concat_dataset = concat_dataset.remove_columns(['__index_level_0__',])
    # concat_dataset.push_to_hub(f'nayohan/koquality_raw')

    calc = DataSelect(df)

    split_len_df_list = calc.make_split_len(args.len)
    np_emb_all_list = calc.make_embedding(calc.make_split_len(args.len), tokenizer, model, device)
    
    for k_val in k_values:
        clustered_df_list = calc.run_clustering(args.len, split_len_df_list, np_emb_all_list, k_val)
        for m_val in m_values:
            for n_val in n_values:
                sampled_df_list = calc.sampling_data(clustered_df_list, m_val, n_val)
                result = calc.concat_df_list(sampled_df_list)

                # 결과를 저장할 디렉토리 생성
                output_dir = f"results_len{args.len}_cluster{k_val}"
                os.makedirs(output_dir, exist_ok=True)

                # 파일 이름 생성
                filename = f"result_len{args.len}_k{k_val}_m{m_val}_n{n_val}.json"
                output_path = os.path.join(output_dir, filename)

                # 데이터프레임을 JSON 파일로 저장
                final_result = pd.DataFrame({"cluster": result['cluster'], "ppl": result['ppl'], "instruction": result['instruction'], "output": result['output']})
                final_result.to_json(output_path, orient='records', lines=True, force_ascii=False)