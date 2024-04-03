### runs a benchmark 

### Experiment parameters
# Defining the model
model_type = "Llama-2" # eg. "Llama-2", "BioLlama"
model_size = "7b" # eg. "7b", "13b", "70b"

# Defining the retrieval component if model is of type BioLlama
if model_type == "BioLlama":
    corpus = "RCT200k" # eg. 
    retriever = "MedCPT"