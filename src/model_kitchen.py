### Utility function to create the required Llama-2 or BioLlama object
### Written by Neel Rajani

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.biollama import BioLlama

def prepare_model_path(
        model_type: str,
        model_size: str,
        model_state: str,
) -> str:
    model_path = "meta-llama/Llama-2-70b-chat-hf"
    return model_path

def model( 
        model_type: str,
        model_size: str,
        model_state: str,
        torch_dtype: torch.dtype,
        RETRO_layer_ids: List[int],
        training: bool,
        retriever_name: str,
        db_name: str,
        neighbour_length: int):
    
    # Identify the correct model path
    model_path = prepare_model_path(
        model_type = model_type,
        model_size = model_size,
        model_state = model_state)
    
    # Create model as specified by model_type
    if model_type == "Llama-2":
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    device_map = "auto",
                                                    torch_dtype = torch_dtype, # Should still be hardcoded to float32
                                                    cache_dir = "../hf_cache/")
        model.tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                                  cache_dir = "../hf_cache/")
    elif model_type == "BioLlama":
        model = BioLlama(
            model_path = model_path,
            torch_dtype = torch_dtype,
            RETRO_layer_ids = RETRO_layer_ids,
            training = False,
            retriever_name = retriever_name,
            db_name = db_name,
            neighbour_length = neighbour_length,
        )       
    return model