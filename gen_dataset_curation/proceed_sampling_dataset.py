import os
import argparse
import pandas as pd

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
                clustered_df = clustered_df.drop_duplicates(['instruction'])
                clustered_ppl_df = clustered_df[:prop]
                clustered_ppl_df_list.append(clustered_ppl_df)
            df = pd.concat(clustered_ppl_df_list, ignore_index=True)

        new_df_list.append(df)
    return new_df_list

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process data using DataSelect class")
    parser.add_argument("--len", type=int, default=10, help="make_split_len_group")
    parser.add_argument("--load_cluster_path", type=str, default='./result_ppl_len', help="save path of the cluster column added to the ppl corpus")
    parser.add_argument("--save_final_path", type=str, default='./results_final', help="save path of the final sampling dataset")

    args = parser.parse_args()
    
    k_values = [1, 10, 20, 50, 100]
    m_values = ['rand', 'ppl', 'ppl_h']
    n_values = [0.01, 0.05, 0.1]

    ppl_cluster_dataset_path = f'{args.load_cluster_path}{args.len}'

    # 4. Sampling by method(M) with sampling ratio (N)
    for k_val in k_values:
        ppl_df = pd.read_json(f'{ppl_cluster_dataset_path}/len{args.len}_k{k_val}.json', lines=True, orient='records')
        ppl_df_bool_list = [ppl_df['len_group']==len for len in range(args.len)] # load len_group
        clustered_df_list = [ppl_df[df_bool] for df_bool in ppl_df_bool_list] 
        
        for m_val in m_values:
            for n_val in n_values:
                sampled_df_list = sampling_data(clustered_df_list, k_val, m_val, n_val)
                result =  pd.concat(sampled_df_list, ignore_index=True)
                result = result.sort_values(by=['len_group', 'cluster', 'ppl', 'len'], axis=0)
                
                # 결과를 저장할 디렉토리 생성
                output_dir = f'{args.save_final_path}_len{args.len}'
                os.makedirs(output_dir, exist_ok=True)
                filename = f"len{args.len}_k{k_val}_m{m_val}_n{n_val}.json"
                output_path = os.path.join(output_dir, filename)
                result.to_json(output_path, orient='records', lines=True, force_ascii=False)