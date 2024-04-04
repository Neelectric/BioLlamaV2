### runs a benchmark 
### Written by Neel Rajani

from src.biollama import BioLlama
from src.retrieval import load_db
from src.run_benchmark import run_benchmark

### Experiment parameters
# Defining the model
model_type = "Llama-2" # eg. "Llama-2", "BioLlama"
model_size = "7b" # eg. "7b", "13b", "70b"
model_state = "untrained" # eg. "untrained", "pre-trained", "fine-tuned"

# Defining the retrieval component if model is of type BioLlama
if model_type == "BioLlama":
    retrieval_corpus = "RCT200k" # eg. "RCT200k", "PMA_y2k", "PMC_y2k"
    retriever = "MedCPT"
    neighbour_length = 32

# Defining the benchmark to use
benchmark_name = "MedMCQA" # eg. "BioASQ5b", "PubMedQA", "MedQA-4", "MedQA-5", "MedMCQA"
benchmark_start_idx = 10
num_questions = 1000

run_benchmark(
        model_type = model_type,
        model_size = model_size,
        model_state = model_state,
        benchmark_name = benchmark_name,
        benchmark_start_idx = benchmark_start_idx,
        num_questions = num_questions,
        retrieval_corpus = retrieval_corpus,
        retriever = retriever,
        neighbour_length = neighbour_length,
        )