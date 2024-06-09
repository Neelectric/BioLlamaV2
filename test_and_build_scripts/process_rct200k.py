### open the txt file in rct200k/train.txt

from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Path: test_and_build_scripts/process_rct200k.py
with open('rct200k/train.txt', 'r') as f:
    lines = f.readlines()


# at the end of the day, what is it i am trying to do here?
# what i really want, is to have a raw abstract_id starting at 0 mapped to an abstract, in a JSON
# we'll go from there
# how do i do that? well, empty lines are the delimiter between abstracts. so we can just split on that

abstracts = {}
current_abstract_id = 0
current_abstract = ""
for line in tqdm(lines):
    # if line starts with ###, skip
    if line.startswith("###"):
        continue
    if line == "\n":
        abstracts[current_abstract_id] = current_abstract
        current_abstract_id += 1
        current_abstract = ""
    else:
        current_abstract += line
print("all abstracts processed.")

# now we need to clean each abstract. any mentions of "\n" and "\t" should be removed
for abstract_id, abstract in abstracts.items():
    abstract = abstract.replace("\n", " ").replace("\t", "").replace("BACKGROUND", "").replace("OBJECTIVE", "").replace("METHODS", "").replace("RESULTS", "").replace("CONCLUSIONS", "")
    #if the string contains "Unique identifier NCT", then chop this off and everything after it
    if "Unique identifier NCT" in abstract:
        abstract = abstract.split("Unique identifier NCT")[0]
        # print("   NCT found")
    if "ACTRN" in abstract:
        abstract = abstract.split("ACTRN")[0]
        # print("   ACTRN found")
    abstracts[abstract_id] = abstract

print("all abstracts cleaned.")

# with open('rct200k/train_cleaned.json', 'w') as f:
#     json.dump(abstracts, f)

article_model_path = "ncbi/MedCPT-Article-Encoder"
article_tokenizer = AutoTokenizer.from_pretrained(article_model_path)
article_model = AutoModel.from_pretrained(
    article_model_path, 
    device_map = "auto",
    # device = "cuda"
    )

llama_path = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, cache_dir = "../hf_cache/")
llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# now we need to tokenize each abstract, split each abstract into chunks of 32 tokens, and save each chunk to an id in a JSON
# we'll also save the abstract_id that the chunk came from
# if the last chunk is less than 32 tokens, we'll skip it
abstract_chunks = {}

# create a list to store the chunks
# create a list to store the chunks and their info
chunks_list = []
chunks_info = []

for abstract_id, abstract in tqdm(abstracts.items()):
    tokens = llama_tokenizer(abstract, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = tokens["input_ids"]
    for i in range(1, input_ids.size(1), 32):
        chunk = input_ids[0, i:i+32]
        # also, if current chunk contains pad tokens, skip it
        if llama_tokenizer.pad_token_id in chunk:
            print(f"skipping chunk {i} in abstract {abstract_id} because it contains pad tokens")
            continue
        if chunk.size(0) == 32:
            detokenized_chunk = llama_tokenizer.decode(chunk)
            chunks_list.append(chunk.unsqueeze(0))  # add an extra dimension to make it a batch
            chunks_info.append((abstract_id, detokenized_chunk))

# define batch size
batch_size = 100  # adjust this based on your GPU memory

# create a list to store all embeddings
embeds_list = []

# process chunks in batches
for i in tqdm(range(0, len(chunks_list), batch_size)):
    batch = torch.cat(chunks_list[i:i+batch_size], 0).to("cuda")  # shape: (batch_size, 32)
    with torch.no_grad():
        embeds = article_model(batch).last_hidden_state[:, 0, :]
    embeds_list.append(embeds.cpu())  # move embeddings to CPU memory and add them to the list
    torch.cuda.empty_cache()  # free up GPU memory

# concatenate all embeddings into a single tensor
embeds_tensor = torch.cat(embeds_list, 0)

# now you can build the index
index, index_infos = build_index(embeds_tensor, save_on_disk=True)

with open('rct200k/train_chunks.json', 'w') as f:
    json.dump(abstract_chunks, f)