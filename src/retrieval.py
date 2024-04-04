### A refactored and cleaned version of db_retrieval.py in BioLlamaV1
### Written by Neel Rajani

from typing import Tuple
import faiss

def load_db(db_name: str, retriever_name: str) -> Tuple[faiss.IndexFlatIP, dict]:
    """
    Args:
        db_name (str): The name of the database to load.
        retriever_name (str): The name of the retriever to use.

    Returns:
        Tuple[faiss.IndexFlatIP, dict]: A tuple containing the loaded FAISS index and JSON object.
    """
    # Load the FAISS index and JSON object
    db_faiss = ...  # Load the FAISS index
    db_json = ...  # Load the JSON object

    return db_faiss, db_json