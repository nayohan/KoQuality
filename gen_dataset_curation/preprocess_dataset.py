## datasets
from datasets import load_dataset
import pandas as pd
from torch.utils.data import DataLoader

ds1 = load_dataset("nayohan/KoAlpaca-v1.1a_ppl",split="train") # koalpaca
df1 = pd.DataFrame({"instruction":ds1['instruction'], "output":ds1['output'], "ppl":ds1['ppl'], "len":ds1['len']})

ds2 = load_dataset("nayohan/kullm-v2_ppl",split="train") # kullm
df2 = pd.DataFrame(ds2)
df2["full_instruction"] = df2["instruction"] + " " + df2['input']
df2 = pd.DataFrame({"instruction":df2['full_instruction'], "output":df2['output'], "ppl":df2['ppl'], "len":df2['len']})

ds3 = load_dataset("nayohan/OIG-small-chip2-ko_ppl",split="train") # kullm
df3 = pd.DataFrame({"instruction":ds3['user_translated'], "output":ds3['chip2_translated'], "ppl":ds3['ppl'], "len":ds3['len']})


def make_dataset():
    all_dfs = [df1, df2, df3]
    df = pd.concat(all_dfs[:1000])  # 데이터프레임 concat

    # '\n'을 공백으로 대체
    df['instruction'] = df['instruction'].apply(lambda x: x.replace('\n', " "))
    return df