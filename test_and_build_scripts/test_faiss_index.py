import faiss
import glob
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import json
import orjson
from tqdm import tqdm
from time import time
import multiprocessing
import threading
from itertools import islice

import dask.bag as db
import json
from dask.distributed import Client


embeddings_dir = '/root/nfs/pubmed_cleaned_embeds/'
index_path = '/root/nfs/pubmed_cleaned_index/'

# load all files under /root/nfs/pubmed_cleaned/ as a list
embeds_files = glob.glob("/root/nfs/pubmed_cleaned/*.tsv")
print(embeds_files)

time_before_index_load = time()
my_index = faiss.read_index(index_path + "medcpt_index.faiss")
time_after_index_load = time()
time_to_load_index = time_after_index_load - time_before_index_load
print(f"Time to load index: {time_to_load_index}")



client = Client()
client.dashboard_link
file_path = index_path + "lookup_table_definitive.json"
time_before_json_load = time()
bag = db.read_text(file_path, blocksize=100 * 1000 * 1000).map(json.loads)

# results = load_json_parallel(file_path, num_processes=num_threads)
time_after_json_load = time()
time_to_load_json = time_after_json_load - time_before_json_load
print(f"Time to load json: {time_to_load_json}")





# model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
# tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

# queries = [
# 	"diabetes treatment", 
# 	"How to treat diabetes?", 
# 	"A 45-year-old man presents with increased thirst and frequent urination over the past 3 months.",
# ]

# with torch.no_grad():
# 	encoded = tokenizer(
# 		queries[0], 
# 		truncation=True, 
# 		padding=True, 
# 		return_tensors='pt', 
# 		max_length=64,
# 	)
	
# 	embeds = model(**encoded).last_hidden_state[:, 0, :]
# 	print(embeds.size())

# time_after_encoding = time()
# time_to_encode = time_after_encoding - time_after_index_load
# print(f"Time to encode: {time_to_encode}")

# k = 5
# distances, indices = my_index.search(embeds, k)
# time_after_search = time()
# time_to_search = time_after_search - time_after_encoding
# print(f"Time to search: {time_to_search}")

# # temp = my_index.search_level_0(embeds, k) # missing 5 required positional arguments: 'k', 'nearest', 'nearest_d', 'distances', and 'labels'
# # temp3 = my_index.search_c(embeds,k) # missing 3 required positional arguments: 'k', 'distances', and 'labels'
# # temp4 = my_index.search_and_reconstruct(embeds, k)
# # temp5 = my_index.range_search(embeds, k)

# # print(list(zip(distances[0], indices[0])))

# with open('/root/nfs/pubmed_cleaned_index/lookup_table.json', "r") as file:
# 	lookup_table = json.load(file)
	
# neighbors = []
# for index in indices[0]:
# 	print(index)
# 	index = str(index)
# 	print(lookup_table[index])
# 	neighbors.append(lookup_table[index])
	
# llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# tokenized = llama2_tokenizer(neighbors)
# for neighbor in tokenized:
# 	print(neighbor)