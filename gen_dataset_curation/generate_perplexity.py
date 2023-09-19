import os
import argparse
import evaluate
import pandas as pd

import torch
from tqdm import tqdm

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM


def calc_ppl(model, encodings, stride=256):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    max_length = model.config.max_position_embeddings
    encodings['input_ids'] = torch.tensor(encodings['input_ids'])
    seq_len = encodings['input_ids'].size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings['input_ids'][:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


def calc_perplextiy_upload(model_name, dataset_name, save_local=True, save_local_path="./result_ppl", save_hf=False, save_hf_user_name=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token':'<|endoftext|>'})

    dataset = load_dataset(dataset_name)

    def encode_pad_preprocess(examples):
        return tokenizer(examples['instruction'], max_length=1024, truncation=True, return_tensors='pt')

    dataset = load_dataset(dataset_name)
    preprocessed_data = dataset.map(encode_pad_preprocess, num_proc=24)
    print(preprocessed_data)
    
    ppl_results = []
    for data in tqdm(preprocessed_data['train']):
        ppl = calc_ppl(model, data)
        ppl_results.append(ppl.detach().cpu().tolist())
    round_ppl_results = [round(ppl, 2) for ppl in  ppl_results]#ppl_results['perplexities']]

    df = pd.DataFrame(preprocessed_data['train'])
    df['ppl'] = round_ppl_results
    df_dataset = pd.DataFrame(df.sort_values(by=['len', 'ppl']), columns=['len', 'ppl', 'group', 'instruction', 'output'])
    df_dataset = df_dataset.dropna(axis=0)
    
    datasetname = dataset_name.split('/')[-1]
    model_name = model_name.split('/')[-1]
    
    if save_local and save_local_path: # save to local folder
        os.makedirs(save_local_path, exist_ok=True)
        df_dataset.to_json(f"./{save_local_path}/{datasetname}_ppl_{model_name}.json", orient='records', lines=True, force_ascii=False)

    if save_hf and save_hf_user_name: # push to hub
        df_dataset = pd.DataFrame(df_dataset, columns=['instruction', 'output', 'group', 'len', 'ppl'])

        dataset = Dataset.from_pandas(df_dataset)
        dataset = DatasetDict({'train': dataset})
        df_dataset = df_dataset.remove_columns(['__index_level_0__',])
        dataset.push_to_hub(f'{save_hf_user_name}/{datasetname}_ppl_instruction_{model_name}')
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process data using DataSelect class")
    parser.add_argument("--device", type=int, default=0, help="gpu device number")
    parser.add_argument("-d", "--dataset_name", type=str, default='beomi/KoAlpaca-v1.1a', help="raw corpus dataset name")
    parser.add_argument("-m", "--model_name", type=str, default='EleutherAI/polyglot-ko-1.3b', help="model name when calc perplexity")
    parser.add_argument("--save_local", type=bool, default=True, help="save path after add ppl column")
    parser.add_argument("--save_ppl_path", type=str, default='./result_ppl', help="save path after add ppl column")
    parser.add_argument("--save_hf", type=bool, default=False, help="save path after add ppl column")
    parser.add_argument("--save_hf_user_name", type=str, default='nayohan', help="raw corpus dataset name")
    args = parser.parse_args()
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # dataset_name = 'beomi/KoAlpaca-v1.1a'
    # dataset_name = 'nlpai-lab/kullm-v2'
    # dataset_name = 'nayohan/koquality_raw_test'
    # model_name = 'gpt2'
    # model_name = 'EleutherAI/polyglot-ko-1.3b'
    
    # 2. calculate perplexity and save to ppl column
    calc_perplextiy_upload(
                            model_name=args.model_name, 
                            dataset_name=args.dataset_name, 
                            save_local=args.save_local, 
                            save_local_path=args.save_ppl_path, 
                            save_hf=args.save_hf, 
                            save_hf_user_name=args.save_hf_user_name
                            )