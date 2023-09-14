import os
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# dataset_name = 'beomi/KoAlpaca-v1.1a'
# dataset_name = 'nlpai-lab/kullm-v2'
# dataset_name = 'heegyu/OIG-small-chip2-ko'
model_name = 'EleutherAI/polyglot-ko-1.3b'
# model_name = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token':'<|endoftext|>'})

dataset = load_dataset(dataset_name)
dataset = dataset.filter(lambda example: len(example['user_translated'])>0, num_proc=24) # filtering under 0 token length

# filtering over 1024 token length
def encode_preprocess(examples):
    return tokenizer(examples['user_translated'])#, padding=True, return_tensors='pt')

# truncate max length, add padding true
def encode_pad_preprocess(examples):
    return tokenizer(examples['user_translated'], max_length=1024, truncation=True, padding=True, return_tensors='pt')

# extract truncated sentence max_length 1024
def decode_process(examples):
    return {'trunc_instruction': tokenizer.decode(examples['input_ids'], skip_special_tokens=True)}

trunc_data = dataset.map(encode_preprocess, batched=True, num_proc=24)
# trunc_data = trunc_data.filter(lambda example: len(example['input_ids'])<1024, num_proc=24)
encode_pad_data = trunc_data.map(encode_pad_preprocess, batched=True, num_proc=24)
preprocessed_data = encode_pad_data.map(decode_process, num_proc=96)
print(preprocessed_data)


# calculate perplexity
perplexity = evaluate.load("perplexity", module_type="metric")
instruction = preprocessed_data['train']['trunc_instruction']


len_instruction = [len(text) for text in instruction]
ppl_results = perplexity.compute(model_id=model_name, add_start_token=False, predictions=instruction)
round_ppl_results = [round(ppl, 2) for ppl in  ppl_results['perplexities']]

dict_ppl_instruction = dict(zip(instruction, round_ppl_results))
df_dataset = pd.DataFrame(sorted(dict_ppl_instruction.items(), key=lambda x: x[0]), columns=['instruction', 'ppl'])
datasetname = dataset_name.split('/')[-1]
df_dataset.to_json(f"{datasetname}_ppl.json", orient='records', lines=True, force_ascii=False)


# push to huggingface dataset
from datasets import Dataset, DatasetDict
dataset = preprocessed_data['train'].add_column("ppl", round_ppl_results)
dataset = dataset.add_column("len", len_instruction)
dataset = dataset.remove_columns(['input_ids', 'attention_mask', 'trunc_instruction',])

datasetname = dataset_name.split('/')[-1]
model_name = model_name.split('/')[-1]
dataset.push_to_hub(f'nayohan/{datasetname}_ppl_{model_name}')