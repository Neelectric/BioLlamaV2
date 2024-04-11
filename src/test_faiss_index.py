import faiss
import glob
import numpy as np

embeddings_dir = '/root/nfs/pubmed_cleaned_embeds/'
index_path = '/root/nfs/pubmed_cleaned_index/'

my_index = faiss.read_index(glob.glob("my_index_folder/*.index")[0])

query_vector = np.float32(np.random.rand(1, 100))
k = 5
distances, indices = my_index.search(query_vector, k)

print(list(zip(distances[0], indices[0])))