from autofaiss import build_index

embeddings_dir = '/root/nfs/pubmed_cleaned_embeds/'
index_path = '/root/nfs/pubmed_cleaned_index/'

build_index(embeddings=embeddings_dir,
            index_path=index_path + "medcpt_index.faiss",
            index_infos_path=index_path + "index_infos.json")
