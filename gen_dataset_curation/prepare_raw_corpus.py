import argparse
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

def make_dataset():
    # koalpaca-v1.1
    ds1 = load_dataset("beomi/KoAlpaca-v1.1a",split="train") 
    df1 = pd.DataFrame({"instruction":ds1['instruction'], "output":ds1['output'], "group":'koalpaca'})

    # kullm-v2
    ds2 = load_dataset("nlpai-lab/kullm-v2",split="train") 
    df2 = pd.DataFrame(ds2)
    df2["full_instruction"] = df2["instruction"] + " " + df2['input']
    df2['group'] = [text.split('_')[0] for text in df2['id']]
    df2 = pd.DataFrame({"instruction":df2['full_instruction'], "output":df2['output'], "group":df2['group']})

    # oig-small-chip2-ko
    ds3 = load_dataset("heegyu/OIG-small-chip2-ko",split="train") 
    df3 = pd.DataFrame({"instruction":ds3['user_translated'], "output":ds3['chip2_translated'], "group":'oig'})

    all_dfs = [df1, df2, df3]
    df = pd.concat(all_dfs, ignore_index=True)
    
    df['instruction'] = df['instruction'].apply(lambda x: x.strip())#replace('\n', ""))
    df['len'] = [len(text) for text in df['instruction']]
    return df


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process data using DataSelect class")
    parser.add_argument("--save_hf_path", type=str, default='nayohan/koquality_raw_test', help="raw corpus dataset name")
    args = parser.parse_args()
    
    # 1. Dataset Concat 및 Preprocess
    df = make_dataset()
    raw_train = Dataset.from_pandas(df)
    concat_dataset = DatasetDict({'train': raw_train})
    
    # concat_dataset = concat_dataset.remove_columns(['__index_level_0__',])
    concat_dataset = concat_dataset.filter(lambda example: len(example['instruction'])>0)
    concat_dataset = concat_dataset.filter(lambda example: len(example['output'])>0)
    concat_dataset = concat_dataset.sort("len")
    print('concat_dataset:', concat_dataset)
    concat_dataset.push_to_hub(args.save_hf_path)