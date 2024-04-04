### A refactored and cleaned version of biollama.py in BioLlamaV1
### Written by Neel Rajani

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
from retrieval import load_db

def RETRO_fit(biollama, RETRO_layer_ids):
    for i, layer in enumerate(biollama.model.model.layers): # switch pre-specified decoder layers to be a RETRO layer
        if i in RETRO_layer_ids:
            RETROfit_layer(layer, i, biollama, torch_dtype) 
    return

def load_RETRO_weights(biollama, model_path, RETRO_layer_ids):
    return

def new_forward(biollama, *args, **kwargs):
    return

def attach_retriever(biollama, retriever):
    return

class BioLlama():
    """
    The fundamental BioLlama class that holds the modified Llama-2 transformers object and all necessary retrieval components
    """
    def __init__(
            self, 
            model_path: str,
            torch_dtype: torch.dtype,
            RETRO_layer_ids: List[int],
            training: bool,
            retriever_name: str,
            db_name: str,
            neighbour_length: int,
            ):
        
        # Model setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = "../hf_cache/")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir = "../hf_cache/", device_map = "auto", torch_dtype = torch_dtype)
        self.model.generation_config.temperature = 0.01

        # Add RETRO modules and load respective weights if not training
        RETRO_fit(self, RETRO_layer_ids = RETRO_layer_ids)
        if not training: load_RETRO_weights(self, model_path = model_path, RETRO_layer_ids = RETRO_layer_ids)

        # save original .forward() call and then replace .forward() with new customized implementation
        self.model.old_forward = self.model.forward
        self.model.forward = new_forward.__get__(self.model)

        # Prepare all retrieval components, ie. retriever, retrieval corpus and retrieved chunk storage
        attach_retriever(self, retriever = retriever_name)
        self.db_faiss, self.db_json = load_db(db_name = db_name, retriever_name = retriever_name, neighbour_length = neighbour_length)
        self.neighbour_storage = None
        return
    
    def generate():
        return
    
    
        

