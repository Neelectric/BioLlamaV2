### runs a benchmark 
### Written by Neel Rajani

from src.biollama import BioLlama
from src.retrieval import load_db
from src.run_benchmark import run_benchmark

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
else:
    db_name = ""
    retriever_name = ""
    neighbour_length = -1
    RETRO_layer_ids = []

# Defining the benchmark to use
benchmark_name = "BioASQ6b" # eg. "BioASQ6b", "PubMedQA", "MedQA-4", "MedQA-5", "MedMCQA"
benchmark_start_idx = 10
num_questions = 1000

print("running benchmark")
# Run the benchmark
run_benchmark(
        model_type = model_type,
        model_size = model_size,
        model_state = model_state,
        benchmark_name = benchmark_name,
        benchmark_start_idx = benchmark_start_idx,
        num_questions = num_questions,
        db_name = db_name,
        retriever_name = retriever_name,
        neighbour_length = neighbour_length,
        RETRO_layer_ids = RETRO_layer_ids
        )

# Judge the results

# Record the results