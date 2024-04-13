# in a for loop, load in the .npy files in the folder "/nfs/pubmed_cleaned_embeds"

import os
import glob
import torch
import numpy as np
import json


path = '/root/nfs/pubmed_cleaned_embeds/'
embeds = []
for file in os.listdir(path):
    if(file.endswith(".npy")):
        temp = np.load(os.path.join(path, file))

with open('/root/nfs/pubmed_cleaned_index/lookup_table.json', "r") as file:
    lookup_table = json.load(file)

print(len(lookup_table))

def get_file_sizes(directory):
    files = os.listdir(directory)
    source_files = glob.glob("/root/nfs/pubmed_cleaned/*.tsv")
    for source_file in source_files:
        path = os.path.join(directory, source_file)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            size_gb = size / (1024**3)
            print(f"{source_file}: {size_gb:.2f} GB")
            # print(f"{file}: {size} bytes")
directory_path = '/root/nfs/pubmed_cleaned/'
get_file_sizes(directory_path)