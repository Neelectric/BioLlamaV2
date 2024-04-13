### A refactored and cleaned version of db_retrieval.py in BioLlamaV1
### Written by Neel Rajani

from typing import Tuple
import faiss
from time import time
import json

def load_db(db_name: str, retriever_name: str, neighbour_length: int) -> Tuple[faiss.IndexFlatIP, dict]:
    """
    Args:
        db_name (str): The name of the database to load.
        retriever_name (str): The name of the retriever to use.

    Returns:
        Tuple[faiss.IndexFlatIP, dict]: A tuple containing the loaded FAISS index and JSON object.
    """

    index_path = '/root/nfs/pubmed_cleaned_index/'
    time_before_index_load = time()
    db_faiss = faiss.read_index(index_path + retriever_name + "_index.faiss")
    time_after_index_load = time()
    time_to_load_index = time_after_index_load - time_before_index_load
    print(f"Time to load index: {time_to_load_index}")

    with open("/root/nfs/pubmed_cleaned_index/lookup_table_" + retriever_name + ".json", "r") as file:
        db_json = json.load(file)
    return db_faiss, db_json