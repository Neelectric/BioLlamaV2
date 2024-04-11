### A minor script to build a faiss index from local tsv files
### Written by Neel Rajani, with Copilot active for file I/O

from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
from tqdm import tqdm


# Helper function to return 32 token chunks from prompt
def split_into_chunks(text, max_chunk_length=32):
    tokens = llama2_tokenizer.encode(text)
    # inputs = llama2_tokenizer(text, return_tensors="pt", padding=False)
    chunks = []
    if len(tokens) < max_chunk_length:
         return [text]
    for i in range(0, len(tokens), max_chunk_length):
         if (i+max_chunk_length) <= len(tokens):
            chunk = tokens[i:i+max_chunk_length]
            decoded = llama2_tokenizer.decode(chunk, skip_special_tokens=True)
            chunks.append(decoded)
    return chunks

llama_path = "meta-llama/Llama-2-7b-chat-hf"
medcpt_path = "ncbi/MedCPT-Article-Encoder"
llama2_tokenizer = AutoTokenizer.from_pretrained(llama_path, cache_dir = "../hf_cache/")
document_tokenizer = AutoTokenizer.from_pretrained(medcpt_path)
document_model = AutoModel.from_pretrained(medcpt_path).to('cuda')
embeds_dir = '/root/nfs/pubmed_cleaned_embeds'
os.makedirs(embeds_dir, exist_ok=True)


tsv_basename = os.path.basename('/root/nfs/pubmed_cleaned/abs_1_0.tsv').split(".")[0]
embeds_for_file = []
batch_size = 128

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

with open('/root/nfs/pubmed_cleaned/abs_1_0.tsv', "r") as f:
    print(sum(bl.count("\n") for bl in blocks(f)))


with open('/root/nfs/pubmed_cleaned/abs_1_0.tsv', 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines):
            abstract = line.strip()
            chunks = split_into_chunks(abstract)
            with torch.no_grad(): # Code taken directly from MedCPT github
                encoded = document_tokenizer(
                    chunks, 
                    truncation=True, 
                    padding=True, 
                    return_tensors='pt', 
                    max_length=512,
                ).to("cuda")
                embeds = document_model(**encoded).last_hidden_state[:, 0, :] 
                embeds_for_file.extend(embeds.cpu().numpy())
embeds_path = os.path.join(embeds_dir, f"{tsv_basename}.npy")
np.save(embeds_path, np.array(embeds_for_file))
