print("started script")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print("imported...")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map = "auto")

print("model loaded!")

prompt = "In the era of generative AI, "
tokenized_prompt = tokenizer.encode(prompt, return_tensors = "pt").to('cuda')
raw_output = model.generate(tokenized_prompt,
                            max_new_tokens = 50,
                            temperature = 0.01)
untokenized_output = tokenizer.decode(raw_output[0], skip_special_tokens = True)
print(untokenized_output)