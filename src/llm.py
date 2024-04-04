### Utility function to create the required Llama-2 or BioLlama object
### Written by Neel Rajani

from biollama import BioLlama
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def llm(self, 
        model_path: str,
        torch_dtype: torch.dtype,
        RETRO_layer_ids: List[int],
        training: bool,
        retriever_name: str,
        db_name: str,
        neighbour_length: int,)