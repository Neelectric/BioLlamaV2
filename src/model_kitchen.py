### Utility function to create the required Llama-2 or BioLlama object
### Written by Neel Rajani

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from biollama import BioLlama

def prepare_model_path(
        model_type: str,
        model_size: str,
        model_state: str,
) -> str:
    model_path = "meta-llama/Llama-2-13b-chat-hf"
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
    
    

    return