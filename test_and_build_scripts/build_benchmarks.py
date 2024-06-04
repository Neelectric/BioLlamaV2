from datasets import load_dataset
from time import time


time_before = time()
MedMCQA = load_dataset("openlifescienceai/medmcqa")
MedQA_4 = load_dataset("GBaker/MedQA-USMLE-4-options")
MedQA_5 = load_dataset("bigbio/med_qa")
BioASQ6b = load_dataset("HiTZ/Multilingual-BioASQ-6B", 'en')
PubMedQA = load_dataset("qiaojin/PubMedQA", "pqa_labeled") # "pqa_artifical"
time_after = time()

# print(MedQA_5["train"][0])
print(f"Time taken to load the dataset: {time_after - time_before} seconds")