import os
import copy
import argparse

import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from datasets import Dataset, DatasetDict

class SentenceEmbeddingClustering:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data using DataSelect class")
    parser.add_argument("--len", type=int, default=10, help="make_split_len_group")
    parser.add_argument("--device", type=int, default=0, help="gpu device number")
    parser.add_argument("-d", "--dataset_name", type=str, default='beomi/KoAlpaca-v1.1a', help="raw corpus dataset name")
    parser.add_argument("-m", "--model_name", type=str, default='EleutherAI/polyglot-ko-1.3b', help="model name when calc perplexity")
    parser.add_argument("-e", "--emb_model_name", type=str, default=True, help="model name when make sentence embedding")
    parser.add_argument("--load_ppl_path", type=str, default='./result_ppl', help="ppl added corpus load path")
    parser.add_argument("--save_cluster_path", type=str, default='./result_ppl_len', help="save path of the cluster column added to the ppl corpus")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    k_values = [1, 10, 20, 50, 100]
    m_values = ['rand', 'ppl', 'ppl_h']
    n_values = [0.01, 0.05, 0.1]

    # 3. generate sentence embedding and length(L) based clustering(K)
    args.datasetname = args.dataset_name.split('/')[-1]
    args.modelname = args.model_name.split('/')[-1]
    ppl_df = pd.read_json(  f"./{args.load_ppl_path}/{args.datasetname}_ppl_{args.modelname}.json", lines=True, orient='records')
    # ppl_df = ppl_df.loc[:10000, :]
    # ppl_df['len_group']=""
    
    calc = SentenceEmbeddingClustering(ppl_df)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModel.from_pretrained(args.emb_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.emb_model_name)
    
    split_len_df_list = calc.make_split_len(args.len)
    np_emb_all_list = calc.make_embedding(calc.make_split_len(args.len), tokenizer, model, device)
    
    for k_val in k_values:
        clustered_df_list = calc.run_clustering(args.len, split_len_df_list, np_emb_all_list, k_val)
        final_df = pd.concat(clustered_df_list, ignore_index=True)
        final_df = final_df.sort_values(by=['len_group', 'cluster', 'ppl', 'len'], axis=0)
        final_df = final_df[["ppl", "len", "len_group", "cluster", "group", "instruction", "output"]]

        output_dir = f"{args.save_cluster_path}{args.len}"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"len{args.len}_k{k_val}.json"
        output_path = os.path.join(output_dir, filename)
        final_df.to_json(output_path, orient='records', lines=True, force_ascii=False)

        # # upload to huggingface dataset
        # raw_train = Dataset.from_pandas(final_df)
        # final_dataset = DatasetDict({'train': raw_train})
        # final_dataset = final_dataset.remove_columns(['ppl'])
        # final_dataset.push_to_hub(f'nayohan/koquality_len{args.len}_k{k_val}')