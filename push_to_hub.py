from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/polyglot-ko-5.8b_len5_k100_mppl_n0.01_bs16')
# model.push_to_hub("nayohan/polyglot-ko-5.8b-Inst")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/polyglot-ko-5.8b_len5_k100_mppl_n0.01_bs16')
# tokenizer.push_to_hub("nayohan/polyglot-ko-5.8b-Inst")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/polyglot-1.3b-len10_k100_n01')
# model.push_to_hub("nayohan/polyglot-ko-1.3b-Inst", revision="Len10_K100_N0.1")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/polyglot-1.3b-len10_k100_n01')
# tokenizer.push_to_hub("nayohan/polyglot-ko-1.3b-Inst", revision="Len10_K100_N0.1")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/polyglot-5.8b-len10_k100_n01')
# model.push_to_hub("nayohan/polyglot-ko-5.8b-Inst-v1.1")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/polyglot-5.8b-len10_k100_n01')
# tokenizer.push_to_hub("nayohan/polyglot-ko-5.8b-Inst-v1.1")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/polyglot-5.8b-koquality_raw')
# model.push_to_hub("nayohan/polyglot-ko-5.8b-Inst-v1.2")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/polyglot-5.8b-koquality_raw')
# tokenizer.push_to_hub("nayohan/polyglot-ko-5.8b-Inst-v1.2")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/llama-2-ko-7b-len10_k100_mppl_n0.1')
# model.push_to_hub("nayohan/llama-2-ko-7b-Inst")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/llama-2-ko-7b-len10_k100_mppl_n0.1')
# tokenizer.push_to_hub("nayohan/llama-2-ko-7b-Inst")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/ko-ref-llama2-7b-len10_k100_mppl_n0.1')
# model.push_to_hub("nayohan/ko-ref-llama2-7b-Inst")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/ko-ref-llama2-7b-len10_k100_mppl_n0.1')
# tokenizer.push_to_hub("nayohan/ko-ref-llama2-7b-Inst")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/ko-ref-llama2-7b-len10_k100_mppl_n0.1')
# model.push_to_hub("nayohan/ko-ref-llama2-7b-Inst")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/ko-ref-llama2-7b-len10_k100_mppl_n0.1')
# tokenizer.push_to_hub("nayohan/ko-ref-llama2-7b-Inst")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/KoQuality-polyglot-ko-1.3b')
# model.push_to_hub("DILAB-HYU/KoQuality-Polyglot-1.3b")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/KoQuality-polyglot-ko-1.3b')
# tokenizer.push_to_hub("DILAB-HYU/KoQuality-Polyglot-1.3b")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/KoQuality-polyglot-ko-3.8b')
# model.push_to_hub("DILAB-HYU/KoQuality-Polyglot-3.8b")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/KoQuality-polyglot-ko-3.8b')
# tokenizer.push_to_hub("DILAB-HYU/KoQuality-Polyglot-3.8b")

# model = AutoModelForCausalLM.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/KoQuality-ko-ref-llama2-7b')
# model.push_to_hub("DILAB-HYU/KoQuality-ko-ref-llama2-7b")
# tokenizer = AutoTokenizer.from_pretrained('/home/uj-user/Yo/HiT5/HCLT/train_llm/KoQuality-ko-ref-llama2-7b')
# tokenizer.push_to_hub("DILAB-HYU/KoQuality-ko-ref-llama2-7b")