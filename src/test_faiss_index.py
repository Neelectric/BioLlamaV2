import faiss
import glob
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import json

embeddings_dir = '/root/nfs/pubmed_cleaned_embeds/'
index_path = '/root/nfs/pubmed_cleaned_index/'

my_index = faiss.read_index(index_path + "medcpt_index.faiss")

model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

queries = [
	"diabetes treatment", 
	"How to treat diabetes?", 
	"A 45-year-old man presents with increased thirst and frequent urination over the past 3 months.",
]

with torch.no_grad():
	encoded = tokenizer(
		queries[0], 
		truncation=True, 
		padding=True, 
		return_tensors='pt', 
		max_length=64,
	)
	
	embeds = model(**encoded).last_hidden_state[:, 0, :]
	print(embeds)
	print(embeds.size())

k = 5
distances, indices = my_index.search(embeds, k)
# temp = my_index.search_level_0(embeds, k) # missing 5 required positional arguments: 'k', 'nearest', 'nearest_d', 'distances', and 'labels'
# temp3 = my_index.search_c(embeds,k) # missing 3 required positional arguments: 'k', 'distances', and 'labels'
# temp4 = my_index.search_and_reconstruct(embeds, k)
# temp5 = my_index.range_search(embeds, k)

print(list(zip(distances[0], indices[0])))

with open('/root/nfs/pubmed_cleaned_index/lookup_table.json', "r") as file:
	lookup_table = json.load(file)
	
for index in indices:
	print(index)
	index = int(index)
	print(lookup_table[index])