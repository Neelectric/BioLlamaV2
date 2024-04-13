### A minor script to build a faiss index from local tsv files
### Written by Neel Rajani, with Copilot active for file I/O

from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
from tqdm import tqdm
import json
import glob

# Helper function to return 32 token chunks from prompt
def split_into_chunks(text, max_chunk_length=32):
    tokens = llama2_tokenizer.encode(text)
    # inputs = llama2_tokenizer(text, return_tensors="pt", padding=False)
    chunks = []
    if len(tokens) < max_chunk_length:
         return []
    for i in range(0, len(tokens), max_chunk_length):
         if (i+max_chunk_length) <= len(tokens):
            chunk = tokens[i:i+max_chunk_length]
            decoded = llama2_tokenizer.decode(chunk, skip_special_tokens=True) # crucial to not add "<s>"
            chunks.append(decoded)
    return chunks

llama_path = "meta-llama/Llama-2-7b-chat-hf"
medcpt_path = "ncbi/MedCPT-Article-Encoder"
llama2_tokenizer = AutoTokenizer.from_pretrained(llama_path, cache_dir = "../hf_cache/")
document_tokenizer = AutoTokenizer.from_pretrained(medcpt_path)
document_model = AutoModel.from_pretrained(medcpt_path).to('cuda')
embeds_dir = '/root/nfs/pubmed_cleaned_embeds'
os.makedirs(embeds_dir, exist_ok=True)


# tsv_basename = os.path.basename('/root/nfs/pubmed_cleaned/abs_1_0.tsv').split(".")[0]

# def blocks(files, size=65536):
#     while True:
#         b = files.read(size)
#         if not b: break
#         yield b

# with open('/root/nfs/pubmed_cleaned/abs_1_0.tsv', "r") as f:
#     print(sum(bl.count("\n") for bl in blocks(f)))

source_files = glob.glob("/root/nfs/pubmed_cleaned/*.tsv")
lookup_table = {}
lookup_count = 0
file_count = 0

for source_file in tqdm(source_files):
    # print(f"Processing {source_file}")
    batch_size = 128
    all_embeddings = []
    all_chunks = []
    tsv_basename = os.path.basename(source_file).split(".")[0]
    if file_count == 2:
        break

    with open(source_file, 'r') as file:
        all_chunks = []
        lines = file.readlines()
        for start_idx in tqdm(range(0, len(lines), batch_size)):
            end_idx = min(start_idx + batch_size, len(lines))
            batch_abstracts = [line.strip() for line in lines[start_idx:end_idx]]
            batch_chunks = [split_into_chunks(abstract) for abstract in batch_abstracts]
            for abstract in batch_abstracts:
                chunks = split_into_chunks(abstract)
                if chunks != []:
                    all_chunks += chunks
            
        tokens = document_tokenizer(all_chunks, padding=True, return_tensors="pt")
        input_ids = tokens.input_ids.to("cuda")
        step_size = 50
        for i in tqdm(range(0, len(input_ids), step_size)):
            temp_input_ids = input_ids[i:i+step_size]
            with torch.no_grad():
                embeds = document_model(temp_input_ids).last_hidden_state[:, 0, :]
                all_embeddings.extend(embeds.cpu().numpy())

    embeds_path = os.path.join(embeds_dir, f"{tsv_basename}.npy")
    np.save(embeds_path, np.array(all_embeddings))

    for (chunk, embed) in zip(all_chunks,all_embeddings):
        lookup_table[lookup_count] = chunk
        lookup_count += 1
    file_count += 1
with open('/root/nfs/pubmed_cleaned_index/lookup_table.json', 'w') as json_file:
    json.dump(lookup_table, json_file)