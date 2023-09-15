import os
import evaluate
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer

    
def calc_perplextiy_upload(model_name, dataset_name, save_local=True, save_local_path="./result_ppl", save_hf=False, save_hf_user_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token':'<|endoftext|>'})

    dataset = load_dataset(dataset_name)
    dataset = dataset.filter(lambda example: len(example['instruction'])>0, num_proc=24) # filtering under 0 token length

    # filtering over 1024 token length
    def encode_preprocess(examples):
        return tokenizer(examples['instruction'])

    # truncate max length, add padding true
    def encode_pad_preprocess(examples):
        return tokenizer(examples['instruction'], max_length=256, truncation=True, padding=True, return_tensors='pt')

    # extract truncated sentence max_length 1024
    def decode_process(examples):
        return {'trunc_instruction': tokenizer.decode(examples['input_ids'], skip_special_tokens=True)}

    trunc_data = dataset.map(encode_preprocess, batched=True, num_proc=24)
    encode_pad_data = trunc_data.map(encode_pad_preprocess, batched=True, num_proc=24)
    preprocessed_data = encode_pad_data.map(decode_process, num_proc=96)
    print(preprocessed_data)

    # calculate perplexity
    perplexity = evaluate.load("perplexity", module_type="metric")
    instruction = preprocessed_data['train']['trunc_instruction']

    # update perplexity
    ppl_results = perplexity.compute(model_id=model_name, add_start_token=False, predictions=instruction)
    round_ppl_results = [round(ppl, 2) for ppl in  ppl_results['perplexities']]

    df = pd.DataFrame(preprocessed_data['train'])
    df['ppl'] = round_ppl_results
    df_dataset = pd.DataFrame(df.sort_values(by=['len', 'ppl']), columns=['len', 'ppl', 'group', 'instruction', 'output'])
    
    datasetname = dataset_name.split('/')[-1]
    model_name = model_name.split('/')[-1]
    
    if save_local and save_local_path: # save to local folder
        os.makedirs(save_local_path, exist_ok=True)
        df_dataset.to_json(f"./{save_local_path}/{datasetname}_ppl_{model_name}.json", orient='records', lines=True, force_ascii=False)

    if save_hf and save_hf_user_name: # push to hub
        dataset = preprocessed_data['train'].add_column("ppl", round_ppl_results)
        dataset = dataset.remove_columns(['input_ids', 'attention_mask', 'trunc_instruction',])
        # model_name = model_name.split('/')[-1]
        dataset.push_to_hub(f'{save_hf_user_name}/{datasetname}_ppl_instruction_{model_name}')
        
        
if __name__=="__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # dataset_name = 'beomi/KoAlpaca-v1.1a'
    # dataset_name = 'nlpai-lab/kullm-v2'
    dataset_name = 'nayohan/koquality_raw'
    
    # model_name = 'gpt2'
    model_name = 'EleutherAI/polyglot-ko-1.3b'
    
    save_local_path="./result_ppl",
    save_hf_user_name="nayohan"
    
    # 2. calculate perplexity and save to ppl column
    calc_perplextiy_upload(model_name, dataset_name, save_local=True, save_local_path=save_local_path, save_hf=True, save_hf_user_name=save_hf_user_name)