import torch

# Defining the model
model_type = "Llama-2" # eg. "Llama-2", "BioLlama"
model_size = "7b" # eg. "7b", "13b", "70b"
model_state = "untrained" # eg. "untrained", "pre-trained", "fine-tuned"

# If model is of type BioLlama, further define retrieval components and other properties
if model_type == "BioLlama":
    db_name = "RCT200k" # eg. "RCT200k", "PMA_y2k", "PMC_y2k"
    retriever_name = "MedCPT"
    neighbour_length = 32
    RETRO_layer_ids = [15]

# Defining the benchmark to use
benchmark_name = "MedMCQA" # eg. "BioASQ5b", "PubMedQA", "MedQA-4", "MedQA-5", "MedMCQA"
benchmark_start_idx = 10
num_questions = 1000

model_path = "meta-llama/Llama-2-13b-chat-hf"

print("running benchmark")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    device_map = "auto",
                                                    torch_dtype = torch.float32, # Should still be hardcoded to float32
                                                    cache_dir = "../hf_cache/")
model.tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                            cache_dir = "../hf_cache/")