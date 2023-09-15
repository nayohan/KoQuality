## datasets
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

def make_dataset():
    # koalpaca-v1.1
    ds1 = load_dataset("beomi/KoAlpaca-v1.1a",split="train") 
    ds1 = pd.DataFrame(ds1)
    ds1['len'] = [len(text) for text in ds1['instruction']]
    ds1['group'] = 'koalpaca'
    df1 = pd.DataFrame({"instruction":ds1['instruction'], "output":ds1['output'], "len":ds1['len'], "group":ds1['group']})

    # kullm-v2
    ds2 = load_dataset("nayohan/kullm-v2_ppl",split="train") 
    df2 = pd.DataFrame(ds2)
    df2["full_instruction"] = df2["instruction"] + " " + df2['input']
    df2['len'] = [len(text) for text in df2['full_instruction']]
    df2['group'] = [text.split('_')[0] for text in df2['id']]
    df2 = pd.DataFrame({"instruction":df2['full_instruction'], "output":df2['output'], "len":df2['len'], "group":df2['group']})

    # oig-small-chip2-ko
    ds3 = load_dataset("nayohan/OIG-small-chip2-ko_ppl",split="train") 
    df3 = pd.DataFrame(ds3)
    df3['len'] = [len(text) for text in df3['user_translated']]
    df3['group'] = 'oig'
    df3 = pd.DataFrame({"instruction":df3['user_translated'], "output":df3['chip2_translated'], "len":df3['len'], "group":df3['group']})


    all_dfs = [df1, df2, df3]
    df = pd.concat(all_dfs)  # 데이터프레임 concat

    # # '\n'을 공백으로 대체
    # df['instruction'] = df['instruction'].apply(lambda x: x.replace('\n', ""))
    return df


# 1. Dataset Concat 및 Preprocess
df = make_dataset()
raw_train = Dataset.from_pandas(df)
concat_dataset = DatasetDict({'train': raw_train})
concat_dataset = concat_dataset.remove_columns(['__index_level_0__',])
print(concat_dataset)
concat_dataset = concat_dataset.sort("len")#.sort_values(by=['len'], axis=0)
concat_dataset.push_to_hub(f'nayohan/koquality_raw_test')